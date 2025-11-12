# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import (global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow,
                       global_correlation_softmax_stereo, local_correlation_softmax_stereo,
                       correlation_softmax_depth)
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose
from .reg_refine import BasicUpdateBlock
from .utils import normalize_img, feature_add_position, upsample_flow_with_mask
from unimatch.utils.flow_viz import flow_tensor_to_image, flow_to_image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def tensor_img_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) 또는 (H,W,3), float/uint8 모두 허용
    반환: (H,W,3) uint8 RGB
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    # (H,W,3)면 그대로
    if x.ndim == 3 and x.shape[0] in (1,3):  # (C,H,W)
        x = x
        # [-1,1]이면 [0,1]로 맵
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        x = x.permute(1, 2, 0)  # -> (H,W,C)
    elif x.ndim == 3 and x.shape[2] in (1,3):  # (H,W,C)
        x = x
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)
    else:
        raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")

    x = (x.numpy() * 255.0).astype(np.uint8)
    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    return x

class FlowActionAdapter(nn.Module):
    """
    x_cat: (B, 256, H, W)  # x0_last||x1_last
    action: (B, A)
    return: dict with
      - flow_emb_64:   (B, 64)
      - action_emb_64: (B, 64)
      - cond_64:       (B, 64)  # 최종 컨디션
    """
    def __init__(self, in_ch: int = 256, action_dim: int = 7, out_dim: int = 64, mid_dim: int = 128):
        super().__init__()
        # (B,256,H,W) → (B,64)
        self.flow_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # (B,256,1,1)
            nn.Conv2d(in_ch, mid_dim, kernel_size=1),  # (B,128,1,1)
            nn.GELU(),
            nn.Flatten(),                         # (B,128)
            nn.Linear(mid_dim, out_dim),          # (B,64)
            nn.LayerNorm(out_dim)
        )
        # (B,A) → (B,64)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        # (64 + 64) → (64)
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 2, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x_cat: torch.Tensor, action: torch.Tensor) -> dict:
        flow_emb   = self.flow_proj(x_cat)           # (B,64)
        action_emb = self.action_proj(action)        # (B,64)
        fused      = self.fuse(torch.cat([flow_emb, action_emb], dim=-1))  # (B,64)
        return {
            "flow_emb_64": flow_emb,
            "action_emb_64": action_emb,
            "cond_64": fused,
        }


class TinyTokenEncoder(nn.Module):
    """(G, S, D) -> (G, S, D): 단일 Transformer 블록 (Self-Attn + FFN)"""
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (G, S, D)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.ffn(self.norm2(x))
        x = x + h
        return x

class AttnPool2d(nn.Module):
    """
    (B,C,H,W) -> (B, K, D)  # K개 요약 토큰, 차원 D
    learnable queries가 공간 토큰에 MH-Attention으로 풀링
    """
    def __init__(self, in_ch: int, out_dim: int = 64, num_queries: int = 4, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.num_queries = num_queries
        self.q = nn.Parameter(torch.randn(num_queries, out_dim))  # (K, D)
        self.proj_in  = nn.Conv2d(in_ch, out_dim, kernel_size=1)
        self.attn     = nn.MultiheadAttention(embed_dim=out_dim, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, int(out_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(out_dim * mlp_ratio), out_dim),
        )
        nn.init.trunc_normal_(self.q, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj_in(x)                  # (B, D, H, W)
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, D)
        q = self.q.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        pooled, _ = self.attn(query=q, key=tokens, value=tokens)  # (B, K, D)
        pooled = pooled + self.mlp(pooled)  # (B, K, D)
        return pooled

class FlowActionAdapterAttn(nn.Module):
    """
    x_cat: (B, 256, H, W)
    action: (B, A)
    returns:
      - flow_tokens: (B, K, 64)
      - flow_emb_64: (B, 64)  # K개 토큰을 평균(or CLS)로 집계
      - action_emb_64: (B, 64)
      - cond_64: (B, 64)      # 액션을 쿼리로 flow_tokens를 한번 더 집약한 결과
    """
    def __init__(self, in_ch=256, action_dim=7, out_dim=64, mid_dim=128, num_queries=4, heads=4, use_dynamic_common_feature=False, num_dynamic_feature=3, use_linear_prob=False):
        super().__init__()
        self.group_size = num_dynamic_feature

        triplet_mlp_ratio=2.0
        self.attnpool = AttnPool2d(in_ch=in_ch, out_dim=out_dim, num_queries=num_queries, heads=heads)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        # 액션을 쿼리로, flow_tokens를 key/value로 보는 교차어텐션
        self.cross_attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=heads, batch_first=True)
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 2*out_dim),
            nn.GELU(),
            nn.Linear(2*out_dim, out_dim),
        )
        self.use_linear_prob = use_linear_prob
        self.use_dynamic_common_feature=use_dynamic_common_feature
        if self.use_dynamic_common_feature:
            self.triplet_encoder = TinyTokenEncoder(dim=out_dim, heads=heads, mlp_ratio=triplet_mlp_ratio)


    def forward(self, x_cat: torch.Tensor, action: torch.Tensor):
        # 1) 이미지 → K개 요약 토큰(각 64차원)
        flow_tokens = self.attnpool(x_cat)        # (B, K, 64)
        flow_emb_64 = flow_tokens.mean(dim=1)     # 간단 평균 (또는 flow_tokens[:,0]으로 CLS 사용)
        
        # 2) 액션 임베딩
        action_emb = self.action_proj(action)     # (B, 64)

        # 3) 액션을 쿼리(1토큰)로 하여 flow_tokens에서 다시 집약 (액션-조건화)
        q = action_emb.unsqueeze(1)               # (B, 1, 64)
        fused, _ = self.cross_attn(q, flow_tokens, flow_tokens)  # (B,1,64)
        fused = fused + self.cross_ffn(fused)     # (B,1,64)
        cond_64 = fused.squeeze(1)                # (B,64)
        # 4) (B, 64) -> (G, 3, 64)로 reshape 후, triplet feature extraction
        if self.use_dynamic_common_feature or self.use_linear_prob:
            B = cond_64.size(0)
            gs = self.group_size
            if B % gs != 0:
                # 남는 샘플은 잘라내거나(아래) 패딩하도록 바꿔도 됨
                print("Dimension Error!!")
                trim = B - (B // gs) * gs
                if trim > 0:
                    cond_64 = cond_64[:-trim]
                    flow_tokens = flow_tokens[:-trim]
                    flow_emb_64 = flow_emb_64[:-trim]
                    action_emb = action_emb[:-trim]
                B = cond_64.size(0)

            G = B // gs
            cond_triplet = cond_64.view(G, gs, -1)        # (G, 3, 64)
            if self.use_linear_prob and not self.use_dynamic_common_feature:
                cond_64 = cond_triplet.reshape(G, -1)   # (G, 3*64)                
            else:
                # 5) (3,64) 시퀀스에 대해 self-attention 인코딩
                cond_triplet_tokens = self.triplet_encoder(cond_triplet)  # (G, 3, 64)

                # 6) 집계(평균 or 첫 토큰 사용 가능)
                cond_64 = cond_triplet_tokens.mean(dim=1)        # (G, 64)

        return {
            "flow_tokens": flow_tokens,     # (B,K,64) — 정보 보존 ↑
            "flow_emb_64": flow_emb_64,     # (B,64)
            "action_emb_64": action_emb,    # (B,64)
            "cond_64": cond_64,             # (G,64)  ← 최종 컨디션
        }


class Flow2LLaMAAdapter(nn.Module):
    """
    Input : (B, in_ch, H=56, W=56)  e.g., in_ch=256 (x0_last||x1_last)
    Output: (B, 12*12=144, 4096)
    """
    def __init__(self, in_ch=256, mid_ch1=384, mid_ch2=512, out_hw=12, out_dim=4096,
                 gn_groups=32, hidden=1024, use_final_ln=False):
        super().__init__()
        self.out_hw = out_hw
        self.use_final_ln = use_final_ln

        # 56x56 -> 28x28 -> 14x14, 채널 확장
        self.expand = nn.Conv2d(in_ch, mid_ch1, kernel_size=1, bias=False)
        self.gn0 = nn.GroupNorm(gn_groups, mid_ch1)
        self.ds1  = nn.Conv2d(mid_ch1, mid_ch1, kernel_size=3, stride=2, padding=1, bias=False)  # 56->28
        self.gn1  = nn.GroupNorm(gn_groups, mid_ch1)
        self.ds2  = nn.Conv2d(mid_ch1, mid_ch2, kernel_size=3, stride=2, padding=1, bias=False)  # 28->14
        self.gn2  = nn.GroupNorm(gn_groups, mid_ch2)

        # 14x14 -> 12x12 (안정적인 축소)
        self.pool = nn.AdaptiveAvgPool2d((out_hw, out_hw))

        # 로컬 문맥 한번 더 섞기(선택적이지만 성능에 도움)
        self.dw = nn.Conv2d(mid_ch2, mid_ch2, kernel_size=3, padding=1, groups=mid_ch2, bias=False)
        self.gn3 = nn.GroupNorm(gn_groups, mid_ch2)

        # MLP: mid_ch2 -> 4096 (토큰 임베딩)
        self.proj1 = nn.Linear(mid_ch2, hidden)
        self.proj2 = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()
        if use_final_ln:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):                     # x: (B,256,56,56)
        B = x.size(0)
        x = self.act(self.gn0(self.expand(x)))        # (B,384,56,56)
        x = self.act(self.gn1(self.ds1(x)))           # (B,384,28,28)
        x = self.act(self.gn2(self.ds2(x)))           # (B,512,14,14)
        x = self.pool(x)                               # (B,512,12,12)
        x = self.act(self.gn3(self.dw(x)))            # (B,512,12,12)

        # (B,512,12,12) -> (B,144,512) -> (B,144,4096)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, self.out_hw*self.out_hw, -1)
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        if self.use_final_ln:
            x = self.ln(x)
        return x   # (B,144,4096)

class FlowActionAdapter(nn.Module):
    def __init__(self, c_flow=2, c_feat=128, n_heads=4, num_dynamic_feature: int = 3):
        super().__init__()
        self.flow_enc = nn.Sequential(
            nn.Conv2d(c_flow, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, c_feat, 3, padding=1)
        )
        # pos_pe를 '버퍼'로 한 번만 등록
        self.register_buffer("pos_pe", None, persistent=False)  # torch>=1.9에서 None 허용

        self.action_mlp = nn.Sequential(
            nn.Linear(7, 256), nn.GELU(),
            nn.Linear(256, c_feat)
        )
        self.group_size=num_dynamic_feature
        self.attn = nn.MultiheadAttention(embed_dim=c_feat, num_heads=n_heads, batch_first=True)
        self.proj = nn.Linear(c_feat, 64)

    def _positional_enc(self, H, W, C, device, dtype):
        # 새로 만들어야 하는 조건
        need_new = (
            self.pos_pe is None
            or self.pos_pe.shape[1] != H * W
            or self.pos_pe.shape[2] != C
            or self.pos_pe.device != device
            or self.pos_pe.dtype  != dtype
        )
        if need_new:
            # base: (2, H, W)  with values in [-1, 1]
            y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
            x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            base = torch.stack([yy, xx], dim=0)  # (2, H, W)

            # 채널 확장: (2*k, H, W) >= (C, H, W)
            k = (C + 1) // 2                     # 최소한 C 채널을 넘기도록
            pe = base.repeat(k, 1, 1)            # (2*k, H, W)
            pe = pe[:C]                          # (C, H, W)

            # (1, HW, C)
            pe = pe.reshape(C, H * W).transpose(0, 1).unsqueeze(0)  # (1, HW, C)

            with torch.no_grad():
                self.pos_pe = pe  # register_buffer로 다시 등록하지 말고 값만 교체
        return self.pos_pe


    def forward(self, flow_uv, action):
        B, C2, H, W = flow_uv.shape
        f = self.flow_enc(flow_uv)                 # (B,C,H,W)
        tokens = f.flatten(2).transpose(1, 2)      # (B,HW,C)

        pe = self._positional_enc(H, W, tokens.size(-1), tokens.device, tokens.dtype)
        tokens = tokens + pe

        q = self.action_mlp(action).unsqueeze(1)   # (B,1,C)
        attn_out, _ = self.attn(q, tokens, tokens) # (B,1,C)
        h = attn_out.squeeze(1)                    # (B,C)
        cond_64 = self.proj(h)                     # (B,64)
        B = cond_64.size(0)
        gs = self.group_size
        G = B // gs
        cond_triplet = cond_64.view(G, gs, -1)        # (G, 3, 64)
        cond_64 = cond_triplet.reshape(G, -1)   # (G, 3*64)      
        return {"cond_64": cond_64}


def normalize_flow_to_grid(flow_uv: torch.Tensor) -> torch.Tensor:
    """
    flow_uv: (B, 2, H, W), 단위=픽셀
    반환: (B, 2, H, W), 단위=정규화 좌표계([-1,1])
    """
    B, C, H, W = flow_uv.shape
    assert C == 2, "flow_uv should have 2 channels (u, v)"
    # grid_sample 좌표계에 맞춰 스케일
    # u: [-1,1]는 x축, v: [-1,1]는 y축
    u = flow_uv[:, 0] / ((W - 1) / 2.0)
    v = flow_uv[:, 1] / ((H - 1) / 2.0)
    flow_norm = torch.stack([u, v], dim=1)
    return flow_norm

def zscore_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B, C, H, W) -> 각 샘플/채널별 평균0, 표준편차1
    """
    mean = x.mean(dim=(2,3), keepdim=True)
    std  = x.std(dim=(2,3), keepdim=True).clamp_min(eps)
    return (x - mean) / std

class UniMatchVisionBackbone(nn.Module):
    def __init__(self, base_unimatch: nn.Module, fuse_multiscale: bool = False, use_dynamic_common_feature: bool = False, num_dynamic_feature: int = 3, use_linear_prob: bool=False):
        super().__init__()

        self.base_unimatch=base_unimatch
        self.optical_backbone = base_unimatch.backbone
        self.optical_transformer = base_unimatch.transformer
        self.num_scales = getattr(base_unimatch, "num_scales", 1)
        self.feature_channels = getattr(base_unimatch, "feature_channels", 128)
        self.fuse_multiscale = fuse_multiscale
        self.attn_type = "swin"
        self.attn_splits_list = [2, 8]
        self.feature_dim = 64
        self.use_dynamic_common_feature=use_dynamic_common_feature
        self.num_dynamic_feature=num_dynamic_feature
        self.use_linear_prob=use_linear_prob
        # x0_last (128) || x1_last (128) -> 256 입력을 받아 12x12 토큰(144개) * 4096 임베딩으로
        self.optical_llama_adapter = Flow2LLaMAAdapter(in_ch=self.feature_channels*2,
                                               mid_ch1=384, mid_ch2=512,
                                               out_hw=12, out_dim=4096,
                                               gn_groups=32, hidden=1024,
                                               use_final_ln=False)
        self.flow_action_adapter = FlowActionAdapterAttn(
            in_ch=self.feature_channels * 2,  # 128*2=256
            action_dim=7,
            out_dim=self.feature_dim,
            mid_dim=self.feature_dim*2,
            use_dynamic_common_feature=self.use_dynamic_common_feature,
            num_dynamic_feature=self.num_dynamic_feature,
            use_linear_prob=self.use_linear_prob,
        )
        self.flow_action = FlowActionAdapter(
            num_dynamic_feature=self.num_dynamic_feature
        )
        if self.use_linear_prob:
            if self.use_dynamic_common_feature:
                in_dim = self.feature_dim
            else:
                in_dim = self.feature_dim * self.num_dynamic_feature
            mlp_hidden = 256
            self.cls_mlp = nn.Sequential(        
                nn.Linear(in_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.GELU(),
            )
            self.cls_head=nn.Linear(mlp_hidden, 3)
        
        self._freeze_all_except_flow_action()

    def tokens_from_pooled(self, x, target_tokens=128):
        B,C,H,W = x.shape
        r = H / W
        h = max(1, int((target_tokens * r) ** 0.5))
        w = max(1, target_tokens // h)
        y = F.adaptive_avg_pool2d(x, (h, w))
        return y.permute(0,2,3,1).reshape(B, h*w, C)

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)
        features = self.optical_backbone(concat)      # list of [2B, C, H, W], hi->lo
        features = features[::-1]             # lo->hi
        feature0, feature1 = [], []
        for feat in features:
            f0, f1 = torch.chunk(feat, 2, dim=0)
            feature0.append(f0)
            feature1.append(f1)
        return feature0, feature1

    def _freeze_all_except_flow_action(self):
        # 전부 동결
        for p in self.parameters():
            p.requires_grad = False
        # flow_action_adapter만 학습 가능
        # for p in self.flow_action_adapter.parameters():
        #     p.requires_grad = True
        for p in self.flow_action.parameters():
            p.requires_grad = True
        if self.use_linear_prob:
            for p in self.cls_head.parameters():
                p.requires_grad = True
            for p in self.cls_mlp.parameters():
                p.requires_grad = True
        # 러닝 스탯 갱신 방지 (BN/GN 등): eval 모드 고정
        self.optical_backbone.eval()
        self.optical_transformer.eval()
        self.optical_llama_adapter.eval()

    def forward(self, img0, img1, action=None):
        # with torch.no_grad():
        #     img0, img1 = normalize_img(img0, img1)  # [B,3,H,W]
        #     f0_list, f1_list = self.extract_feature(img0, img1)

        #     tokens_per_scale = []
        #     for s in range(self.num_scales):
        #         f0, f1 = f0_list[s], f1_list[s]                  # (B,128,Hs,Ws)
        #         attn_splits = self.attn_splits_list[s] if self.attn_splits_list is not None else 1
        #         f0_pe, f1_pe = feature_add_position(f0, f1, attn_splits, self.feature_channels)
        #         x0, x1 = self.optical_transformer(f0_pe, f1_pe, attn_type=self.attn_type, attn_num_splits=attn_splits)
        #         tokens_per_scale.append((x0, x1))               # (B,128,Hs,Ws)

        #     # 마지막(최고 해상도) 스케일 사용
        #     x0_last, x1_last = tokens_per_scale[-1]             # (B,128,56,56) 가정
        #     x_cat = torch.cat([x0_last, x1_last], dim=1)        # (B,256,56,56)
        with torch.no_grad():
            flow_uv = self.base_unimatch.extract_optical_feature(img0, img1) # B * 2 * 60 * 108
        if action is not None:
            # proj = self.flow_action_adapter(x_cat, action)
            proj = self.flow_action(flow_uv, action)  # cond_64/logits 사용
            if self.use_linear_prob:
                logits = self.cls_head(self.cls_mlp(proj["cond_64"]))
                return logits
            return proj["cond_64"]
            #action : B * 7

        # 56->12 / ch 확장 / 4096 임베딩 / 토큰 144개
        # llama_tokens = self.optical_llama_adapter(x_cat)            # (B,144,4096)

        # return {
            # "tokens_per_scale": tokens_per_scale,
            # "tokens": (x0_last, x1_last),        # 참고용
            # "llama_tokens": llama_tokens,        # LLaMA inputs_embeds로 바로 사용
        # }

        # # 멀티스케일 결합 (권장 2: coarsest + finest)
        # x0_coarse, x1_coarse = tokens_per_scale[0]
        # x0_fine,   x1_fine   = tokens_per_scale[-1]

        # B, C, Hf, Wf = x0_fine.shape
        # x0c_up = F.interpolate(x0_coarse, size=(Hf, Wf), mode="bilinear", align_corners=True)
        # x1c_up = F.interpolate(x1_coarse, size=(Hf, Wf), mode="bilinear", align_corners=True)

        # x0_fused = torch.cat([x0_fine, x0c_up], dim=1)  # (B, 256, Hf, Wf)
        # x1_fused = torch.cat([x1_fine, x1c_up], dim=1)

        # x0_fused = self.merge(x0_fused)  # (B,128,Hf,Wf)
        # x1_fused = self.merge(x1_fused)

        # return {
        #     "tokens_per_scale": tokens_per_scale,
        #     "tokens": (x0_fused, x1_fused),  # flow 정보 풍부한 멀티스케일 토큰
        # }

class UniMatchFlowWDepth(nn.Module):
    def __init__(self, optical_backbone: nn.Module,  use_dynamic_common_feature: bool = False, num_dynamic_feature: int = 3, use_linear_prob: bool=False, load_pretrained_dynamic_model_path: str = None):
        super().__init__()
       
        self.optical_backbone = optical_backbone
        self.feature_channels = getattr(optical_backbone, "feature_channels", 128)
        self.feature_dim = 64
        self.num_dynamic_feature=num_dynamic_feature
        self.use_linear_prob = use_linear_prob
        self.use_dynamic_common_feature=use_dynamic_common_feature
        self.fuse_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),   # 112x112
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),  # 56x56
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)  # [B,128,1,1]
        )
        self.vis_proj = nn.Linear(128, 64)   # 비주얼 64d

        # action(7) → 64d
        self.act_proj = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(),
            nn.Linear(64, 64)
        )

        # proj: (64+64) → 64 (cond_64처럼 쓰기)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64), nn.GELU()
        )

        # 최종 7-class 분류기
        self.classifier = nn.Linear(64, 7)
        self._freeze_all_except_flow_action()
        self.load_pretrained_dynamic_model_path = load_pretrained_dynamic_model_path
        if load_pretrained_dynamic_model_path is not None:
            self.load_pretrained_model_weights(load_pretrained_dynamic_model_path)
            self.freeze_all()
            self.use_linear_prob = False
        
    def load_pretrained_model_weights(self, load_pretrained_dynamic_model_path: str):
        load_checkpoint = torch.load(load_pretrained_dynamic_model_path, weights_only=False)
        self.load_state_dict(load_checkpoint['model'], strict=False)
        print("Loaded pretrained model from", load_pretrained_dynamic_model_path)

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _freeze_all_except_flow_action(self):
        # 전부 동결
        for p in self.optical_backbone.parameters():
            p.requires_grad = False
        if self.use_linear_prob:
            for p in self.classifier.parameters():
                p.requires_grad = True
        self.optical_backbone.eval()
            
    def forward(self, img0, img1, depth_0=None, depth_1=None, action=None, angle=None, viz=False):
        """
        flow_uv:  [B, 2, 224, 224]
        depth_0:  [B, 224, 224]
        depth_1:  [B, 224, 224]
        """
        with torch.no_grad():
            batch_size = img0.size(0)
            flow_uv = self.optical_backbone(
                img0, img1,
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                padding_factor=32,
                num_reg_refine=6,
                attn_type='swin',
                task = 'flow'
                )["flow_preds"][0]
            # flow_uv2 = self.optical_backbone(
            #     img0[batch_size//2:], img1[batch_size//2:],
            #     attn_splits_list=[2, 8],
            #     corr_radius_list=[-1, 4],
            #     prop_radius_list=[-1, 1],
            #     padding_factor=32,
            #     num_reg_refine=6,
            #     attn_type='swin',
            #     task = 'flow'
            #     )["flow_preds"][0]
            # flow_uv = torch.cat([flow_uv, flow_uv2], dim=0)
            # flow_uv = dict["flow_preds"], 총 4개가 나오고, 하나마다 batch_size * 2 * 256 * 256이 나옴
        if  self.load_pretrained_dynamic_model_path is not None:
            with torch.no_grad():
                if viz == True:
                    self.visualize_flow(img0, img1, flow_uv)
                if depth_0 is not None and depth_1 is not None:
                    if depth_0.dim() == 3: depth_0 = depth_0.unsqueeze(1)
                    if depth_1.dim() == 3: depth_1 = depth_1.unsqueeze(1)
                    visual_feature = torch.cat([depth_0, flow_uv, depth_1], dim=1)
                else:
                    visual_feature = flow_uv

                vis_feat = self.fuse_encoder(visual_feature).flatten(1)
                # [B,128] → [B,64]
                vis_64 = self.vis_proj(vis_feat)

                if action is not None:
                    #[B,7] → [B,64]
                    act_64 = self.act_proj(action)
                
                fused_128 = torch.cat([vis_64, act_64], dim=1)
                
                cond_64 = self.proj_head(fused_128)
                if self.use_linear_prob:
                    logits = self.classifier(cond_64)
                else:
                    logits = cond_64
                return logits
        else:
            if viz == True:
                self.visualize_flow(img0, img1, flow_uv)
            if depth_0 is not None and depth_1 is not None:
                if depth_0.dim() == 3: depth_0 = depth_0.unsqueeze(1)
                if depth_1.dim() == 3: depth_1 = depth_1.unsqueeze(1)
                visual_feature = torch.cat([depth_0, flow_uv, depth_1], dim=1)
            else:
                visual_feature = flow_uv

            vis_feat = self.fuse_encoder(visual_feature).flatten(1)
            # [B,128] → [B,64]
            vis_64 = self.vis_proj(vis_feat)

            if action is not None:
                #[B,7] → [B,64]
                act_64 = self.act_proj(action)
            
            fused_128 = torch.cat([vis_64, act_64], dim=1)
            
            cond_64 = self.proj_head(fused_128)
            if self.use_linear_prob:
                logits = self.classifier(cond_64)
            else:
                logits = cond_64
            return logits
        
    def visualize_flow(self, img0, img1, flow_uv):
        for i in range(flow_uv.shape[0]):
            import cv2
            flow_rgb = flow_tensor_to_image(flow_uv[i])  # (H,W,3), BGR, uint8 or float
            cv2.imwrite(f"flow_{i:04d}.png", cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))
            if isinstance(flow_rgb, torch.Tensor):
                flow_rgb = flow_rgb.detach().cpu().numpy()
            if flow_rgb.dtype != np.uint8:
                flow_rgb = (np.clip(flow_rgb, 0, 1) * 255).astype(np.uint8)
            img_flow_rgb = flow_rgb[..., ::-1]  # BGR -> RGB

            # img0, img1: (B,3,H,W) -> (H,W,3) uint8
            img0_rgb = img0[i].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            img1_rgb = img1[i].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

            # 시각화: img0 | img1 | flow
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img0_rgb);  axes[0].set_title(f"img0[{i}]");  axes[0].axis("off")
            axes[1].imshow(img1_rgb);  axes[1].set_title(f"img1[{i}]");  axes[1].axis("off")
            axes[2].imshow(img_flow_rgb); axes[2].set_title(f"flow[{i}]"); axes[2].axis("off")
            plt.tight_layout()
            plt.show()
            cv2.imwrite(f"flow_{i:04d}.png", img_flow_rgb[..., ::-1])  # 다시 RGB->BGR로 저장
            cv2.imwrite(f"img0_{i:04d}.png", img0_rgb[..., ::-1])
            cv2.imwrite(f"img1_{i:04d}.png", img1_rgb[..., ::-1])

class UniMatch(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 task='flow',
                 ):
        super(UniMatch, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        if not self.reg_refine or task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic

        if reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 if task == 'flow' else 1,
                                           bilinear_up=task == 'depth',
                                           )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def forward(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=1,
                pred_bidir_flow=False,
                task='flow',
                intrinsics=None,
                pose=None,  # relative pose transform
                min_depth=1. / 0.5,  # inverse depth range
                max_depth=1. / 10,
                num_depth_candidates=64,
                depth_from_argmax=False,
                pred_bidir_depth=False,
                **kwargs,
                ):
        self.training = False
        if pred_bidir_flow:
            assert task == 'flow'

        if task == 'depth':
            assert self.num_scales == 1  # multi-scale depth model is not supported yet

        results_dict = {}
        flow_preds = []

        if task == 'flow':
            # stereo and depth tasks have normalized img in dataloader
            img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        if task != 'depth':
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            feature0_ori, feature1_ori = feature0, feature1

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if task == 'depth':
                # scale intrinsics
                intrinsics_curr = intrinsics.clone()
                intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / upsample_factor

            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                assert task != 'depth'
                flow = flow.detach()

                if task == 'stereo':
                    # construct flow vector for disparity
                    # flow here is actually disparity
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    # NOTE: reverse disp, disparity is positive
                    displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                    feature1 = flow_warp(feature1, displace)  # [B, C, H, W]
                elif task == 'flow':
                    feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                else:
                    raise NotImplementedError

            attn_splits = attn_splits_list[scale_idx]
            if task != 'depth':
                corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )
            # correlation and softmax
            if task == 'depth':
                # first generate depth candidates
                b, _, h, w = feature0.size()
                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(feature0)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h,
                                                                                               w)  # [B, D, H, W]

                flow_pred = correlation_softmax_depth(feature0, feature1,
                                                      intrinsics_curr,
                                                      pose,
                                                      depth_candidates=depth_candidates,
                                                      depth_from_argmax=depth_from_argmax,
                                                      pred_bidir_depth=pred_bidir_depth,
                                                      )[0]

            else:
                if corr_radius == -1:  # global matching
                    if task == 'flow':
                        flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
                    elif task == 'stereo':
                        flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
                    else:
                        raise NotImplementedError
                else:  # local matching
                    if task == 'flow':
                        flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
                    elif task == 'stereo':
                        flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]
                    else:
                        raise NotImplementedError

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            if task == 'stereo':
                flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if (pred_bidir_flow or pred_bidir_depth) and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation

            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution

                    if task == 'stereo':
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        flow_up_pad = self.upsample_flow(flow_pad, feature0)
                        flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
                    elif task == 'depth':
                        depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                          is_depth=True).clamp(min=min_depth, max=max_depth)
                        flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]
                    else:
                        flow_up = self.upsample_flow(flow, feature0)

                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert num_reg_refine > 0
                    for refine_iter_idx in range(num_reg_refine):
                        flow = flow.detach()

                        if task == 'stereo':
                            zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                        elif task == 'depth':
                            if pred_bidir_depth and refine_iter_idx == 0:
                                intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                                pose = torch.cat((pose, torch.inverse(pose)), dim=0)

                                feature0_ori, feature1_ori = torch.cat((feature0_ori, feature1_ori),
                                                                       dim=0), torch.cat((feature1_ori,
                                                                                          feature0_ori), dim=0)

                            flow_from_depth = compute_flow_with_depth_pose(1. / flow.squeeze(1),
                                                                           intrinsics_curr,
                                                                           extrinsics_rel=pose,
                                                                           )

                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow_from_depth,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        else:
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                                  )

                        if task == 'depth':
                            flow = (flow - residual_flow).clamp(min=min_depth, max=max_depth)
                        else:
                            flow = flow + residual_flow

                        if task == 'stereo':
                            flow = flow.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            if task == 'depth':
                                if refine_iter_idx < num_reg_refine - 1:
                                    # bilinear upsampling
                                    flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                                 upsample_factor=upsample_factor,
                                                                 is_depth=True)
                                else:
                                    # last one convex upsampling
                                    # NOTE: clamp depth due to the zero padding in the unfold in the convex upsampling
                                    # pad depth to 2 channels as flow
                                    depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                                    depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                                      is_depth=True).clamp(min=min_depth,
                                                                                           max=max_depth)
                                    flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]

                            else:
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')

                            flow_preds.append(flow_up)

        if task == 'stereo':
            for i in range(len(flow_preds)):
                flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]

        # convert inverse depth to depth
        if task == 'depth':
            for i in range(len(flow_preds)):
                flow_preds[i] = 1. / flow_preds[i].squeeze(1)  # [B, H, W]
        results_dict.update({'flow_preds': flow_preds})
        return results_dict
    
    def extract_optical_feature(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=1,
                pred_bidir_flow=False,
                task='flow',
                intrinsics=None,
                pose=None,  # relative pose transform
                min_depth=1. / 0.5,  # inverse depth range
                max_depth=1. / 10,
                num_depth_candidates=64,
                depth_from_argmax=False,
                pred_bidir_depth=False,
                **kwargs,
                ):
        self.num_scales = 2
        attn_type='swin'
        attn_splits_list=[2]
        corr_radius_list=[-1]
        prop_radius_list=[-1]
        num_reg_refine=6
        if pred_bidir_flow:
            assert task == 'flow'

        if task == 'depth':
            assert self.num_scales == 1  # multi-scale depth model is not supported yet

        results_dict = {}
        flow_preds = []

        if task == 'flow':
            # stereo and depth tasks have normalized img in dataloader
            img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None
        import pdb; pdb.set_trace()
        if task != 'depth':
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1
        import pdb; pdb.set_trace()
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            feature0_ori, feature1_ori = feature0, feature1

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if task == 'depth':
                # scale intrinsics
                intrinsics_curr = intrinsics.clone()
                intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / upsample_factor

            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                assert task != 'depth'
                flow = flow.detach()

                if task == 'stereo':
                    # construct flow vector for disparity
                    # flow here is actually disparity
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    # NOTE: reverse disp, disparity is positive
                    displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                    feature1 = flow_warp(feature1, displace)  # [B, C, H, W]
                elif task == 'flow':
                    feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                else:
                    raise NotImplementedError

            attn_splits = attn_splits_list[scale_idx]
            if task != 'depth':
                corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            import pdb; pdb.set_trace
            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )
            # correlation and softmax
            if task == 'depth':
                # first generate depth candidates
                b, _, h, w = feature0.size()
                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(feature0)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h,
                                                                                               w)  # [B, D, H, W]

                flow_pred = correlation_softmax_depth(feature0, feature1,
                                                      intrinsics_curr,
                                                      pose,
                                                      depth_candidates=depth_candidates,
                                                      depth_from_argmax=depth_from_argmax,
                                                      pred_bidir_depth=pred_bidir_depth,
                                                      )[0]

            else:
                if corr_radius == -1:  # global matching
                    if task == 'flow':
                        flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
                    elif task == 'stereo':
                        flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
                    else:
                        raise NotImplementedError
                else:  # local matching
                    if task == 'flow':
                        flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
                    elif task == 'stereo':
                        flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]
                    else:
                        raise NotImplementedError

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            if task == 'stereo':
                flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if (pred_bidir_flow or pred_bidir_depth) and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation

            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )
        return flow