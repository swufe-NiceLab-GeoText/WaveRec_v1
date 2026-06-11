import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from argparse import Namespace
from utils_pack.utils import make_coord
import numpy as np


class DWT2D(nn.Module):
    def __init__(self, wave: str = 'haar'):
        super(DWT2D, self).__init__()
        self.wave = wave
        if wave == 'haar' or wave == 'db1':
            self.dec_lo = torch.tensor([0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
            self.dec_hi = torch.tensor([-0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
        elif wave == 'db2':
            self.dec_lo = torch.tensor(
                [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604], dtype=torch.float32)
            self.dec_hi = torch.tensor(
                [-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341],
                dtype=torch.float32)
        elif wave == 'db4':
            self.dec_lo = torch.tensor([
                0.0164642402572777, -0.0414649363919634, -0.0673726031230898,
                0.3861100661683982, 0.8127236353996122, 0.4170051863249895,
                -0.0764885997066838, -0.0221457243978065
            ], dtype=torch.float32)
            self.dec_hi = torch.tensor([
                -0.0221457243978065, 0.0764885997066838, 0.4170051863249895,
                -0.8127236353996122, 0.3861100661683982, 0.0673726031230898,
                -0.0414649363919634, -0.0164642402572777
            ], dtype=torch.float32)
        elif wave == 'sym2':
            self.dec_lo = torch.tensor([
                0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604
            ], dtype=torch.float32)
            self.dec_hi = torch.tensor([
                -0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341
            ], dtype=torch.float32)
        else:
            self.dec_lo = torch.tensor([0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
            self.dec_hi = torch.tensor([-0.7071067811865476, 0.7071067811865476], dtype=torch.float32)

        self.filter_len = len(self.dec_lo)
        self.pad_len = self.filter_len // 2

    def forward(self, x: torch.Tensor) -> tuple:
        B, C, H, W = x.shape

        device = x.device
        dtype = x.dtype

        dec_lo = self.dec_lo.to(device).to(dtype).view(1, 1, -1, 1)
        dec_hi = self.dec_hi.to(device).to(dtype).view(1, 1, -1, 1)

        x_padded = F.pad(x, (0, 0, self.pad_len, self.pad_len), mode='reflect')

        lo_row = F.conv2d(x_padded, dec_lo.expand(C, 1, -1, 1), groups=C, stride=(2, 1))
        hi_row = F.conv2d(x_padded, dec_hi.expand(C, 1, -1, 1), groups=C, stride=(2, 1))

        lo_row_padded = F.pad(lo_row, (self.pad_len, self.pad_len, 0, 0), mode='reflect')
        hi_row_padded = F.pad(hi_row, (self.pad_len, self.pad_len, 0, 0), mode='reflect')

        dec_lo_t = self.dec_lo.to(device).to(dtype).view(1, 1, 1, -1)
        dec_hi_t = self.dec_hi.to(device).to(dtype).view(1, 1, 1, -1)

        LL = F.conv2d(lo_row_padded, dec_lo_t.expand(C, 1, 1, -1), groups=C, stride=(1, 2))
        LH = F.conv2d(lo_row_padded, dec_hi_t.expand(C, 1, 1, -1), groups=C, stride=(1, 2))
        HL = F.conv2d(hi_row_padded, dec_lo_t.expand(C, 1, 1, -1), groups=C, stride=(1, 2))
        HH = F.conv2d(hi_row_padded, dec_hi_t.expand(C, 1, 1, -1), groups=C, stride=(1, 2))

        return LL, LH, HL, HH


class IDWT2D(nn.Module):
    def __init__(self, wave: str = 'haar'):
        super(IDWT2D, self).__init__()
        self.wave = wave
        if wave == 'haar' or wave == 'db1':
            self.rec_lo = torch.tensor([0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
            self.rec_hi = torch.tensor([0.7071067811865476, -0.7071067811865476], dtype=torch.float32)
        elif wave == 'db2':
            self.rec_lo = torch.tensor(
                [-0.1294095225512604, 0.2241438680420134, 0.8365163037378079, 0.4829629131445341], dtype=torch.float32)
            self.rec_hi = torch.tensor(
                [-0.4829629131445341, 0.8365163037378079, -0.2241438680420134, -0.1294095225512604],
                dtype=torch.float32)
        elif wave == 'db4':
            self.rec_lo = torch.tensor([
                -0.0221457243978065, -0.0764885997066838, 0.4170051863249895,
                0.8127236353996122, 0.3861100661683982, -0.0673726031230898,
                -0.0414649363919634, 0.0164642402572777
            ], dtype=torch.float32)
            self.rec_hi = torch.tensor([
                -0.0164642402572777, 0.0414649363919634, -0.0673726031230898,
                -0.3861100661683982, 0.8127236353996122, 0.4170051863249895,
                0.0764885997066838, -0.0221457243978065
            ], dtype=torch.float32)
        elif wave == 'sym2':
            self.rec_lo = torch.tensor(
                [-0.1294095225512604, 0.2241438680420134, 0.8365163037378079, 0.4829629131445341], dtype=torch.float32)
            self.rec_hi = torch.tensor(
                [-0.4829629131445341, 0.8365163037378079, -0.2241438680420134, -0.1294095225512604],
                dtype=torch.float32)
        else:
            self.rec_lo = torch.tensor([0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
            self.rec_hi = torch.tensor([0.7071067811865476, -0.7071067811865476], dtype=torch.float32)

        self.filter_len = len(self.rec_lo)

    def forward(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        B, C, H, W = LL.shape
        device = LL.device
        dtype = LL.dtype

        rec_lo = self.rec_lo.to(device).to(dtype).view(1, 1, 1, -1)
        rec_hi = self.rec_hi.to(device).to(dtype).view(1, 1, 1, -1)

        lo_col = F.conv_transpose2d(LL, rec_lo.expand(C, 1, 1, -1), groups=C, stride=(1, 2),
                                    padding=(0, self.filter_len // 2))
        lo_col = lo_col + F.conv_transpose2d(LH, rec_hi.expand(C, 1, 1, -1), groups=C, stride=(1, 2),
                                             padding=(0, self.filter_len // 2))

        hi_col = F.conv_transpose2d(HL, rec_lo.expand(C, 1, 1, -1), groups=C, stride=(1, 2),
                                    padding=(0, self.filter_len // 2))
        hi_col = hi_col + F.conv_transpose2d(HH, rec_hi.expand(C, 1, 1, -1), groups=C, stride=(1, 2),
                                             padding=(0, self.filter_len // 2))

        rec_lo_t = self.rec_lo.to(device).to(dtype).view(1, 1, -1, 1)
        rec_hi_t = self.rec_hi.to(device).to(dtype).view(1, 1, -1, 1)

        output = F.conv_transpose2d(lo_col, rec_lo_t.expand(C, 1, -1, 1), groups=C, stride=(2, 1),
                                    padding=(self.filter_len // 2, 0))
        output = output + F.conv_transpose2d(hi_col, rec_hi_t.expand(C, 1, -1, 1), groups=C, stride=(2, 1),
                                             padding=(self.filter_len // 2, 0))

        target_h = H * 2
        target_w = W * 2
        output = output[:, :, :target_h, :target_w]

        return output


class ExternalContextEncoderV2(nn.Module):
    """
     V2

    ：
    1. ：
    2. ：
    3. ：
    """

    def __init__(self, ext_dim: int = 12, hidden_dim: int = 64, channels: int = 64):
        super(ExternalContextEncoderV2, self).__init__()

        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ext_encoder = nn.Sequential(
            nn.Linear(ext_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, channels),
            nn.LayerNorm(channels)
        )

    def forward(self, ext_features: torch.Tensor) -> torch.Tensor:
        B = ext_features.shape[0]

        time_feat = ext_features[:, 5:6]
        time_emb = self.time_encoder(time_feat)

        ext_emb = self.ext_encoder(ext_features)

        combined = ext_emb.unsqueeze(1) + time_emb.unsqueeze(1)

        attn_output, _ = self.fusion_attention(combined, combined, combined)

        output = self.output_proj(attn_output.squeeze(1))

        return output


class DualStreamCrossAttention(nn.Module):
    """
     (Dual-Stream Cross-Co-Attention)

    ：
    1. ：
    2. ：
    3. ：，

    ：
    - （、）
    - ：
    - 
    """

    def __init__(self, channels: int, ext_dim: int = 64, num_heads: int = 4):
        super(DualStreamCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.ext_proj = nn.Linear(ext_dim, channels)
        self.road_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.ext_qkv = nn.Linear(channels, channels * 3)
        self.road_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        self.ext_out = nn.Linear(channels, channels)
        self.road_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.fusion_gate_ext = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid()
        )

        self.fusion_gate_road = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, low_freq_feat: torch.Tensor, high_freq_feat: torch.Tensor,
                ext_context: torch.Tensor, road_feat: torch.Tensor) -> tuple:
        B, C, H, W = low_freq_feat.shape

        ext_proj = self.ext_proj(ext_context)
        road_proj = self.road_proj(road_feat)

        ext_qkv = self.ext_qkv(ext_proj).view(B, 3, self.num_heads, self.head_dim)
        ext_q, ext_k, ext_v = ext_qkv[:, 0], ext_qkv[:, 1], ext_qkv[:, 2]

        road_qkv = self.road_qkv(road_proj).view(B, C, 3, H, W)
        road_q, road_k, road_v = road_qkv[:, :, 0], road_qkv[:, :, 1], road_qkv[:, :, 2]

        ext_q = ext_q.unsqueeze(1).expand(-1, H * W, -1, -1)
        ext_k = ext_k.unsqueeze(1).expand(-1, H * W, -1, -1)
        ext_v = ext_v.unsqueeze(1).expand(-1, H * W, -1, -1)

        road_q = road_q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        road_k = road_k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        road_v = road_v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        ext_to_road_attn = torch.matmul(ext_q, road_k.transpose(-2, -1)) * self.scale
        ext_to_road_attn = F.softmax(ext_to_road_attn, dim=-1)
        ext_enhanced = torch.matmul(ext_to_road_attn, road_v)

        road_to_ext_attn = torch.matmul(road_q, ext_k.transpose(-2, -1)) * self.scale
        road_to_ext_attn = F.softmax(road_to_ext_attn, dim=-1)
        road_enhanced = torch.matmul(road_to_ext_attn, ext_v)

        ext_enhanced = ext_enhanced.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
        ext_enhanced = self.ext_out(ext_enhanced)

        road_enhanced = road_enhanced.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        road_enhanced = self.road_out(road_enhanced)

        ext_feat_spatial = ext_enhanced.view(B, H, W, C).permute(0, 3, 1, 2)

        ext_gate_input = torch.cat([low_freq_feat.mean(dim=[2, 3]), ext_feat_spatial.mean(dim=[2, 3])], dim=-1)
        ext_gate = self.fusion_gate_ext(ext_gate_input).view(B, C, 1, 1)
        ext_final = low_freq_feat * (1 - ext_gate) + ext_feat_spatial * ext_gate

        road_gate_input = torch.cat([high_freq_feat, road_enhanced], dim=1)
        road_gate = self.fusion_gate_road(road_gate_input)
        road_final = high_freq_feat * (1 - road_gate) + road_enhanced * road_gate

        return ext_final, road_final


class FrequencyAwareFusion(nn.Module):
    """
    

    ：
    1. ：
    2. ：
    3. ：

    ：
    - 
    - 
    - ，
    """

    def __init__(self, channels: int):
        super(FrequencyAwareFusion, self).__init__()

        self.freq_weight_learner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.residual_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor,
                fused_low: torch.Tensor, fused_high: torch.Tensor) -> torch.Tensor:
        B, C, H, W = LL.shape

        freq_concat = torch.stack([fused_low, fused_low, fused_high, fused_high], dim=1)
        freq_weights = self.freq_weight_learner(freq_concat)

        freq_weights = freq_weights.view(B, 4, C, 1, 1)
        freq_weights = F.softmax(freq_weights, dim=1)

        w_ll, w_lh, w_hl, w_hh = freq_weights[:, 0], freq_weights[:, 1], freq_weights[:, 2], freq_weights[:, 3]

        low_freq_combined = w_ll * LL + w_lh * LH
        high_freq_combined = w_hl * HL + w_hh * HH

        combined = torch.cat([low_freq_combined, high_freq_combined], dim=1)
        fused = self.fusion_conv(combined)

        res_gate_input = torch.cat([fused_low, fused], dim=1)
        res_gate = self.residual_gate(res_gate_input)
        output = fused_low * (1 - res_gate) + fused * res_gate

        return output


class WaveletMultiModalFusionV8Pro(nn.Module):
    """
    V8 Pro ：（）

    ：
    1. ：
    2. ：，
    3. ：

    ：
    - 
    - 
    - 
    """

    def __init__(self, channels: int, ext_dim: int = 12, road_channels: int = 64,
                 num_heads: int = 4, dropout: float = 0.1, wave: str = 'haar'):
        super(WaveletMultiModalFusionV8Pro, self).__init__()

        self.channels = channels
        self.wave = wave

        self.dwt = DWT2D(wave=wave)
        self.idwt = IDWT2D(wave=wave)

        self.low_freq_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.high_freq_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.ext_encoder = ExternalContextEncoderV2(ext_dim=ext_dim, hidden_dim=64, channels=channels)

        self.road_encoder = nn.Sequential(
            nn.Conv2d(1, road_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(road_channels),
            nn.GELU(),
            nn.Conv2d(road_channels, road_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(road_channels),
            nn.GELU(),
            nn.Conv2d(road_channels, channels, kernel_size=1)
        )

        self.dual_stream_attention = DualStreamCrossAttention(
            channels=channels,
            ext_dim=channels,
            num_heads=num_heads
        )

        self.freq_aware_fusion = FrequencyAwareFusion(channels)

        self.low_freq_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.high_freq_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.output_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, ext_features: torch.Tensor, road_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        try:
            if H % 2 != 0 or W % 2 != 0:
                pad_h = (2 - H % 2) % 2
                pad_w = (2 - W % 2) % 2
                x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                x_padded = x
                pad_h, pad_w = 0, 0

            LL, LH, HL, HH = self.dwt(x_padded)

            ext_context = self.ext_encoder(ext_features)

            road_feat = self.road_encoder(road_map)
            if road_feat.shape[-2:] != LL.shape[-2:]:
                road_feat = F.interpolate(road_feat, size=LL.shape[-2:], mode='bilinear', align_corners=False)

            LL_proj = self.low_freq_proj(LL)
            LH_proj = self.high_freq_proj(LH)
            HL_proj = self.high_freq_proj(HL)
            HH_proj = self.high_freq_proj(HH)

            LL_fused, LH_fused = self.dual_stream_attention(
                LL_proj, LH_proj, ext_context, road_feat
            )

            HL_fused, HH_fused = self.dual_stream_attention(
                HL_proj, HH_proj, ext_context, road_feat
            )

            LL_refined = self.low_freq_refine(LL_fused)
            LH_refined = self.high_freq_refine(LH_fused)
            HL_refined = self.high_freq_refine(HL_fused)
            HH_refined = self.high_freq_refine(HH_fused)

            fused_freq = self.freq_aware_fusion(
                LL_refined, LH_refined, HL_refined, HH_refined,
                LL_refined, LH_refined + HL_refined + HH_refined
            )

            high_freq_recon = self.idwt(
                fused_freq,
                LH_refined, HL_refined, HH_refined
            )

            if pad_h > 0 or pad_w > 0:
                high_freq_recon = high_freq_recon[:, :, :H // 2, :W // 2]

            low_freq_up = F.interpolate(fused_freq, size=(H, W), mode='bilinear', align_corners=False)
            high_freq_up = F.interpolate(high_freq_recon, size=(H, W), mode='bilinear', align_corners=False)

        except Exception as e:
            low_freq_up = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            high_freq_up = x - low_freq_up

        output = self.output_conv(torch.cat([low_freq_up, high_freq_up], dim=1))

        return output + x


class ECA(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class RDB(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4,
                 use_attention: bool = True, residual_scale: float = 0.1):
        super(RDB, self).__init__()

        self.residual_scale = residual_scale
        self.use_attention = use_attention
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                    nn.GroupNorm(8, growth_rate),
                    nn.GELU(),
                    nn.Dropout2d(0.1)
                )
            )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1),
            nn.GroupNorm(8, in_channels)
        )

        if use_attention:
            self.attention = ECA(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        features = [x]

        for layer in self.layers:
            input_feat = torch.cat(features, dim=1)
            output = layer(input_feat)
            features.append(output)

        fused = self.fusion(torch.cat(features, dim=1))

        if self.use_attention:
            fused = self.attention(fused)

        return residual + fused * self.residual_scale


class DualAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(DualAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.channel_weight = nn.Parameter(torch.ones(1))
        self.spatial_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weights = self.channel_attention(x)
        channel_out = x * channel_weights

        spatial_weights = self.spatial_attention(x)
        spatial_out = x * spatial_weights

        alpha = torch.sigmoid(self.channel_weight)
        beta = torch.sigmoid(self.spatial_weight)
        total_weight = alpha + beta + 1e-6

        return (alpha * channel_out + beta * spatial_out) / total_weight


class EnhancedFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_scales: int = 4):
        super(EnhancedFPN, self).__init__()

        self.num_scales = num_scales

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for _ in range(num_scales)
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])

        self.attention_weights = nn.Parameter(torch.ones(num_scales))

        self.dual_attention_modules = nn.ModuleList([
            DualAttention(out_channels, reduction_ratio=8)
            for _ in range(num_scales)
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * num_scales, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i in range(self.num_scales):
            if i == 0:
                feat = x
            else:
                scale_factor = 2 ** i
                feat = F.adaptive_avg_pool2d(x, (x.shape[2] // scale_factor, x.shape[3] // scale_factor))

            lateral_feat = self.lateral_convs[i](feat)
            features.append(lateral_feat)

        for i in range(len(features) - 1, 0, -1):
            up_feat = F.interpolate(features[i], size=features[i - 1].shape[-2:], mode='bilinear', align_corners=False)
            features[i - 1] = features[i - 1] + up_feat

        enhanced_features = []
        for i, (feat, fpn_conv, dual_attn) in enumerate(zip(features, self.fpn_convs, self.dual_attention_modules)):
            fpn_feat = fpn_conv(feat)
            attended_feat = dual_attn(fpn_feat)

            if i > 0:
                attended_feat = F.interpolate(attended_feat, size=features[0].shape[-2:], mode='bilinear',
                                              align_corners=False)

            enhanced_features.append(attended_feat)

        weighted_features = []
        for feat, weight in zip(enhanced_features, self.attention_weights):
            weighted_feat = feat * torch.sigmoid(weight)
            weighted_features.append(weighted_feat)

        fused_features = torch.cat(weighted_features, dim=1)
        return self.fusion_conv(fused_features)


class Local_Global_Block(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4, num_scales: int = 4):
        super(Local_Global_Block, self).__init__()

        self.local_path = RDB(in_channels, growth_rate, num_layers)
        self.global_path = EnhancedFPN(in_channels, in_channels, num_scales=num_scales)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_hierarchical_feat = self.local_path(x)
        global_multiscale_feat = self.global_path(x)
        fused_feat = torch.cat([local_hierarchical_feat, global_multiscale_feat], dim=1)
        return self.fusion_conv(fused_feat)


class LightweightFeatureExtractorV8Pro(nn.Module):
    """
    V8 Pro 
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64, base_channels: int = 64,
                 num_blocks: int = 1, growth_rate: int = 32, num_layers: int = 4, num_scales: int = 4,
                 ext_dim: int = 12, road_channels: int = 64, use_wavelet: bool = True, wave: str = 'haar'):
        super(LightweightFeatureExtractorV8Pro, self).__init__()

        self.use_wavelet = use_wavelet

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        if use_wavelet:
            self.wavelet_fusion = WaveletMultiModalFusionV8Pro(
                channels=base_channels,
                ext_dim=ext_dim,
                road_channels=road_channels,
                num_heads=4,
                dropout=0.3,
                wave=wave
            )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(Local_Global_Block(base_channels, growth_rate, num_layers, num_scales))

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, ext_features: torch.Tensor, road_map: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        if self.use_wavelet:
            x = self.wavelet_fusion(x, ext_features, road_map)

        for block in self.blocks:
            x = block(x)

        return self.out_conv(x)


class IC_layer(nn.Module):
    def __init__(self, n_channel, drop_rate):
        super(IC_layer, self).__init__()
        self.batch_norm = nn.BatchNorm2d(n_channel)
        self.drop_rate = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.drop_rate(x)
        return x


class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class mini_model(nn.Module):
    def __init__(self, n_channel, scale_factor, in_channel, kernel_size, padding, groups):
        super(mini_model, self).__init__()
        self.n_channels = n_channel
        self.in_channel = in_channel

        self.conv1 = nn.Conv2d(in_channel, n_channel // 2, kernel_size, 1, padding, groups=groups)
        self.bn1 = nn.BatchNorm2d(n_channel // 2)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(n_channel // 2, n_channel, 3, 1, padding=2, dilation=2, groups=groups)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        self.eca = ECALayer(n_channel)
        self.ic_layer = IC_layer(n_channel, 0.3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.eca(x)
        x = self.ic_layer(x)
        return x


class SpatialMask(nn.Module):
    def __init__(self, mask_ratio=0.75, patch_size=8):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "patch_size"

        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        num_patches = x_patch.size(1)
        num_keep = int(num_patches * (1 - self.mask_ratio))

        rand_indices = torch.rand(B, num_patches, device=x.device).argsort(dim=1)
        mask = torch.zeros(B, num_patches, device=x.device)
        mask[:, :num_keep] = 1
        mask = torch.gather(mask, 1, rand_indices.argsort(1))

        x_masked = x_patch * mask.unsqueeze(-1)

        x_masked = rearrange(x_masked, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                             h=H // self.patch_size, w=W // self.patch_size,
                             p1=self.patch_size, p2=self.patch_size)
        return x_masked, mask


class SpatialDecoder(nn.Module):
    def __init__(self, embed_dim=128, scale_factor=4):
        super().__init__()
        self.recon_head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.recon_head(x)


class WaveRec(nn.Module):
    """
    WaveRec V8 Pro: 

    ：
    1. ：
    2. ：，
    3. ：

    ：
    - 
    - 
    - 
    """

    def __init__(self, height=32, width=32, use_exf=True, scale_factor=4,
                 channels=128, sub_region=4, scaler_X=1, scaler_Y=1, args=None,
                 use_wavelet: bool = True, wave: str = 'haar',
                 ext_dim: int = 12, road_channels: int = 64):
        super(WaveRec, self).__init__()
        self.height = height
        self.width = width
        self.masker = SpatialMask(mask_ratio=0.75, patch_size=8)
        self.decoder = SpatialDecoder(embed_dim=1)
        self.fg_height = height * scale_factor
        self.fg_width = width * scale_factor

        self.use_exf = use_exf
        self.n_channels = channels
        self.scale_factor = scale_factor
        self.out_channel = 1
        self.sub_region = sub_region
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.args = args

        self.ic_layer = IC_layer(64, 0.3)

        self.feature_extractor = LightweightFeatureExtractorV8Pro(
            in_channels=1,
            out_channels=64,
            base_channels=128,
            num_blocks=2,
            growth_rate=32,
            num_layers=4,
            num_scales=4,
            ext_dim=ext_dim,
            road_channels=road_channels,
            use_wavelet=use_wavelet,
            wave=wave
        )

        time_span = 15

        if use_exf:
            self.time_emb_region = nn.Embedding(time_span, self.sub_region ** 2)
            self.time_emb_global = nn.Embedding(time_span, (self.fg_width * self.fg_height))

            self.embed_day = nn.Embedding(8, 2)
            self.embed_hour = nn.Embedding(24, 3)
            self.embed_weather = nn.Embedding(128, 3)

            self.ext2lr = nn.Sequential(
                nn.Linear(ext_dim, 64),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.sub_region ** 2),
                nn.ReLU(inplace=True)
            )

            self.ext2lr_global = nn.Sequential(
                nn.Linear(ext_dim, 64),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(64, int(self.fg_width * self.fg_height)),
                nn.ReLU(inplace=True)
            )

            self.global_model = mini_model(self.n_channels, self.scale_factor, 64, 9, 4, 1)
            self.local_sub_model = mini_model(self.n_channels * (int(self.fg_height / self.sub_region) ** 2),
                                              self.scale_factor, 64 * (int(self.fg_height / self.sub_region) ** 2), 3,
                                              1, int(self.fg_height / self.sub_region) ** 2)
        else:
            self.global_model = mini_model(self.n_channels, self.scale_factor, 64, 9, 4, 1)
            self.local_sub_model = mini_model(self.n_channels * (sub_region ** 2),
                                              self.scale_factor, 1024, 3, 1, sub_region ** 2)

        self.relu = nn.ReLU()
        time_conv = []
        for i in range(time_span):
            time_conv.append(nn.Conv2d(256, self.out_channel, 3, 1, 1))
        self.time_conv = nn.Sequential(*time_conv)

        self.time_my = nn.Conv2d(256, 1, 3, 1, 1)

    def embed_ext(self, ext):
        day_idx = ext[:, 4].long().clamp(0, 7).view(-1, 1)
        hour_idx = ext[:, 5].long().clamp(0, 23).view(-1, 1)
        weather_idx = ext[:, 6].long().clamp(0, 127).view(-1, 1)

        ext_out1 = self.embed_day(day_idx).view(-1, 2)
        ext_out2 = self.embed_hour(hour_idx).view(-1, 3)
        ext_out3 = self.embed_weather(weather_idx).view(-1, 3)
        ext_out4 = ext[:, :4]

        return torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], dim=1)

    def normalization(self, x, save_x):
        w = (nn.AvgPool2d(self.scale_factor)(x)) * self.scale_factor ** 2
        w = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')(w)
        w = torch.divide(x, w + 1e-7)
        up_c = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')(save_x)
        x = torch.multiply(w, up_c)
        return x

    def forward(self, x, eif, road_map, is_pretrain=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_yuan = x

        if is_pretrain:
            x_masked, mask = self.masker(x)
            x = x_masked

        x = x.to(device)

        ext_emb = self.embed_ext(eif) if self.use_exf else None

        y = self.feature_extractor(x, ext_emb, road_map)
        x = y

        save_x = x

        coor_hr = make_coord([self.height * self.scale_factor, self.width * self.scale_factor],
                             flatten=False).to(device).unsqueeze(0).expand(x.shape[0], self.height * self.scale_factor,
                                                                           self.width * self.scale_factor, 2)

        if self.use_exf:
            x = self.relu(nn.functional.grid_sample(x, coor_hr.flip(-1), mode='bilinear', align_corners=False))
            x = self.ic_layer(x)
            global_x = x

            x = rearrange(x, 'b c (ph h) (pw w) -> (ph pw) b c h w',
                          ph=int(self.fg_height / self.sub_region),
                          pw=int(self.fg_width / self.sub_region))

            t = eif[:, 5].long().view(-1, 1)
            if self.args.dataset == 'TaxiBJ':
                t -= 7
            t = t.clamp(0, 14)
            #time_emb_region = self.time_emb_region(t).view(-1, 1, int(self.sub_region), int(self.sub_region))
            #time_emb_global = self.time_emb_global(t).view(-1, 1, self.fg_height, self.fg_width)

            #ext_out = self.ext2lr(ext_emb).view(-1, 1, int(self.sub_region), int(self.sub_region))
            #ext_out_global = self.ext2lr_global(ext_emb).view(-1, 1, self.fg_width, self.fg_height)

            output_x = list(map(lambda x: torch.cat([x], dim=1).unsqueeze(0), x))
            output_x = torch.cat(output_x, dim=0)

            local_c = rearrange(output_x, '(ph pw) b c h w -> b (ph pw c) h w',
                                ph=int(self.fg_height / self.sub_region), pw=int(self.fg_width / self.sub_region))

            output = self.local_sub_model(local_c)

            local_f = rearrange(output, 'b (ph pw c) h w -> b c (ph h) (pw w)',
                                ph=int(self.fg_height / self.sub_region), pw=int(self.fg_width / self.sub_region))

            #expanded_map = road_map.repeat(global_x.shape[0], 1, 1, 1)
            global_f = self.global_model(global_x)
        else:
            x = self.relu(nn.functional.grid_sample(x, coor_hr.flip(-1), mode='bilinear', align_corners=False))
            x = self.ic_layer(x)
            global_x = x

            local_c = rearrange(x, 'b c (ph h) (pw w) -> b (ph pw c) h w',
                                ph=self.sub_region, pw=self.sub_region)
            output = self.local_sub_model(local_c)
            local_f = rearrange(output, 'b (ph pw c) h w -> b c (ph h) (pw w)',
                                ph=self.sub_region, pw=self.sub_region)
            global_f = self.global_model(save_x)

        x = torch.cat([local_f, global_f], dim=1)

        output = []
        if self.use_exf:
            for i in range(x.size(0)):
                t = int(eif[i, 5].cpu().detach().numpy())
                if self.args.dataset == 'TaxiBJ':
                    t -= 7
                t = max(0, min(t, 14))
                output.append(self.relu(self.time_conv[t](x[i].unsqueeze(0))))
        else:
            for i in range(x.size(0)):
                output.append(self.relu(self.time_my(x[i].unsqueeze(0))))
        x = torch.cat(output, dim=0)

        x = self.normalization(x, x_yuan * self.scaler_X / self.scaler_Y)

        if is_pretrain:
            sr_output = self.decoder(x)
            return x, sr_output, mask
        else:
            return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = Namespace(dataset='TaxiBJ')

    model = WaveRec(
        height=32,
        width=32,
        use_exf=True,
        scale_factor=4,
        channels=128,
        sub_region=4,
        use_wavelet=True,
        wave='haar',
        ext_dim=12,
        road_channels=64,
        args=args
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")

    dummy_input = torch.randn(2, 1, 32, 32, device=device)
    dummy_ext = torch.randn(2, 7, device=device)
    dummy_road = torch.randn(2, 1, 128, 128, device=device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input, dummy_ext, dummy_road)
        print(f"Forward pass successful! Output shape: {output.shape}")
