import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from argparse import Namespace
from utils_pack.utils import make_coord
import numpy as np

# Reuse all unmodified components from original model
from WaveRec_lightweight_v9Pro import (
    DWT2D, IDWT2D,
    ExternalContextEncoderV2,
    DualStreamCrossAttention,
    ECA, RDB, DualAttention, EnhancedFPN,
    Local_Global_Block,
    IC_layer, ECALayer, mini_model,
    SpatialMask, SpatialDecoder
)


class WaveletMultiModalFusionV2(nn.Module):
    """

    Reserved parameters (for ablation):
    - use_cross_modal: whether to use dual-stream cross-co-attention
    - use_freq_ensemble: whether to use frequency band fusion (baseline is True)
    """

    def __init__(self, channels: int, ext_dim: int = 12, road_channels: int = 64,
                 num_heads: int = 4, dropout: float = 0.1, wave: str = 'haar',
                 use_cross_modal: bool = True, use_freq_ensemble: bool = True):
        super(WaveletMultiModalFusionV2, self).__init__()

        self.channels = channels
        self.wave = wave
        self.use_cross_modal = use_cross_modal
        self.use_freq_ensemble = use_freq_ensemble

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

        if use_cross_modal:
            self.dual_stream_attention = DualStreamCrossAttention(
                channels=channels,
                ext_dim=channels,
                num_heads=num_heads
            )

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

            if self.use_cross_modal:
                # Full version: use dual-stream cross-co-attention
                LL_fused, LH_fused = self.dual_stream_attention(
                    LL_proj, LH_proj, ext_context, road_feat
                )
                HL_fused, HH_fused = self.dual_stream_attention(
                    HL_proj, HH_proj, ext_context, road_feat
                )
            else:
                # Ablation version: no external-road bidirectional attention
                # Use simple addition for multi-modal fusion
                H_f, W_f = LL_proj.shape[2], LL_proj.shape[3]
                ext_spatial = ext_context.view(B, C, 1, 1).expand(-1, -1, H_f, W_f)
                road_resized = road_feat if road_feat.shape[-2:] == (H_f, W_f) else F.interpolate(
                    road_feat, size=(H_f, W_f), mode='bilinear', align_corners=False
                )
                # Simple weighted addition instead of complex cross attention
                LL_fused = LL_proj + ext_spatial * 0.3 + road_resized * 0.3
                LH_fused = LH_proj + ext_spatial * 0.3 + road_resized * 0.3
                HL_fused = HL_proj + ext_spatial * 0.3 + road_resized * 0.3
                HH_fused = HH_proj + ext_spatial * 0.3 + road_resized * 0.3

            LL_refined = self.low_freq_refine(LL_fused)
            LH_refined = self.high_freq_refine(LH_fused)
            HL_refined = self.high_freq_refine(HL_fused)
            HH_refined = self.high_freq_refine(HH_fused)

            # ============================================================
            # Improvement: direct average fusion (replace FrequencyAwareFusion)
            # ============================================================
            if self.use_freq_ensemble:
                # Baseline: direct average of four frequency bands
                fused_freq = (LL_refined + LH_refined + HL_refined + HH_refined) / 4.0
            else:
                # Ablation: remove frequency band fusion, only use LL sub-band
                fused_freq = LL_refined

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


class LightweightFeatureExtractorV2(nn.Module):
    """
    V2 Improved Feature Extractor

    Uses WaveletMultiModalFusionV2 to replace V8Pro version
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64, base_channels: int = 64,
                 num_blocks: int = 1, growth_rate: int = 32, num_layers: int = 4, num_scales: int = 4,
                 ext_dim: int = 12, road_channels: int = 64, use_wavelet: bool = True, wave: str = 'haar',
                 use_cross_modal: bool = True, use_freq_ensemble: bool = True):
        super(LightweightFeatureExtractorV2, self).__init__()

        self.use_wavelet = use_wavelet

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        if use_wavelet:
            self.wavelet_fusion = WaveletMultiModalFusionV2(
                channels=base_channels,
                ext_dim=ext_dim,
                road_channels=road_channels,
                num_heads=4,
                dropout=0.3,
                wave=wave,
                use_cross_modal=use_cross_modal,
                use_freq_ensemble=use_freq_ensemble
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


class WaveRec(nn.Module):
    """
    WaveRec V2 Improved Version

    Reserved ablation parameters:
    - use_wavelet: whether to use wavelet multi-modal fusion
    - use_cross_modal: whether to use dual-stream cross-co-attention
    - use_freq_ensemble: whether to use frequency band fusion (baseline is True, can be set to False for ablation)
    """

    def __init__(self, height=32, width=32, use_exf=True, scale_factor=4,
                 channels=128, sub_region=4, scaler_X=1, scaler_Y=1, args=None,
                 use_wavelet: bool = True, wave: str = 'haar',
                 use_cross_modal: bool = True, use_freq_ensemble: bool = True,
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

        self.feature_extractor = LightweightFeatureExtractorV2(
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
            wave=wave,
            use_cross_modal=use_cross_modal,
            use_freq_ensemble=use_freq_ensemble
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

            output_x = list(map(lambda x: torch.cat([x], dim=1).unsqueeze(0), x))
            output_x = torch.cat(output_x, dim=0)

            local_c = rearrange(output_x, '(ph pw) b c h w -> b (ph pw c) h w',
                                ph=int(self.fg_height / self.sub_region), pw=int(self.fg_width / self.sub_region))

            output = self.local_sub_model(local_c)

            local_f = rearrange(output, 'b (ph pw c) h w -> b c (ph h) (pw w)',
                                ph=int(self.fg_height / self.sub_region), pw=int(self.fg_width / self.sub_region))

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
