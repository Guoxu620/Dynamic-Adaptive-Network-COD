# -*- coding: utf-8 -*-
"""
DABNet Variant: ONLY Module A advanced (DASM_V2 + AD-BiFPN_V2).
Module B is baseline (CBAM + ASPP refine). Includes C toggles (top-k / Gumbel) for DASM.

Forward returns: (stage_pred_list, final_pred_logits_up_to_input_size)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvt import pvt_v2_b2

# ==============================
# Utils & stability
# ==============================
_DEF_EPS = 1e-6
_LOG_EPS = 1e-6
_PROB_MIN = 1e-4
_TAU_MIN = 1e-2
_GUMBEL_MIN = 1e-6
_GUMBEL_MAX = 1.0 - 1e-6

def GN(c, groups: int = 8):
    g = min(groups, c)
    while g > 1 and (c % g != 0):
        g -= 1
    return nn.GroupNorm(num_groups=max(1, g), num_channels=c)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias, stride=stride)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.gn = GN(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x); x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x

# morphology helpers for prompts
def dilate01(x, k=3): return F.max_pool2d(x, kernel_size=k, stride=1, padding=k//2)
def erode01(x, k=3):  return 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=k//2)

# ==============================
# Deformable conv with fallback
# ==============================
try:
    from torchvision.ops import DeformConv2d
    _HAS_DEFORM = True
except Exception:
    DeformConv2d = None
    _HAS_DEFORM = False

class DeformableConvBlockV2(nn.Module):
    def __init__(self, c_in, c_out, with_offset: bool = True, gn=True, guide_channels=1):
        super().__init__()
        self.with_offset = with_offset and _HAS_DEFORM
        mid = c_in
        self.gn1 = GN(mid) if gn else nn.Identity()
        if self.with_offset:
            self.offset = nn.Conv2d(c_in + guide_channels, 18, 3, padding=1)
            self.dcn = DeformConv2d(c_in, mid, kernel_size=3, padding=1, bias=False)
        else:
            self.dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(mid, c_out, 1, bias=False)
        self.gn2 = GN(c_out) if gn else nn.Identity()
        self.act = nn.SiLU(True)
    def forward(self, x, guide: torch.Tensor = None):
        if self.with_offset:
            if guide is None:
                guide = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            off = self.offset(torch.cat([x, guide], dim=1))
            y = self.dcn(x, off)
        else:
            y = self.dw(x)
        y = self.gn1(y); y = self.pw(y); y = self.gn2(y)
        return self.act(y)

# ==============================
# Module A: DASM_V2 (with C toggles)
# ==============================
class DASM_V2(nn.Module):
    def __init__(self, dim_in, dim_out, ks=(3,3,5,3), dil=(1,2,1,1), topk=2, tau_init=1.0, use_gumbel=True, gn_groups=8):
        super().__init__()
        self.pre = nn.Conv2d(dim_in, dim_out, 1, bias=False)
        self.branches = nn.ModuleList()
        for k, d in zip(ks, dil):
            if k == 3 and d == 1 and _HAS_DEFORM:
                self.branches.append(DeformableConvBlockV2(dim_out, dim_out, with_offset=True))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(dim_out, dim_out, k, padding=(k//2)*d, dilation=d, groups=dim_out, bias=False),
                    nn.Conv2d(dim_out, dim_out, 1, bias=False),
                    GN(dim_out, gn_groups), nn.SiLU(True)
                ))
        self.post = nn.Conv2d(dim_out, dim_out, 1, bias=False)
        self.gn   = GN(dim_out, gn_groups)
        self.act  = nn.SiLU(True)
        self.score = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim_out, len(ks), 1, bias=True))
        self.topk = topk
        self.tau  = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        self.use_gumbel = use_gumbel

    @staticmethod
    def _gumbel_like(x):
        u = torch.rand_like(x, dtype=torch.float32).clamp_(min=_GUMBEL_MIN, max=_GUMBEL_MAX)
        return -torch.log(-torch.log(u))

    def _apply_topk(self, probs):
        if (self.topk is None) or (self.topk >= probs.size(1)): return probs
        topk_vals, topk_idx = torch.topk(probs, self.topk, dim=1)
        mask = torch.zeros_like(probs).scatter_(1, topk_idx, 1.0)
        probs = probs * mask
        denom = probs.sum(dim=1, keepdim=True)
        return probs / (denom + _DEF_EPS)

    def forward(self, x):
        x  = self.pre(x)
        B,C,H,W = x.shape
        feats = torch.stack([b(x) for b in self.branches], dim=1)  # [B,K,C,H,W]
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            raw = self.score(x32).view(B, -1, 1, 1, 1)
            tau = torch.clamp(self.tau.abs(), min=_TAU_MIN)
            if self.training and self.use_gumbel:
                g = self._gumbel_like(raw)
                logits = (raw + g) / tau
            else:
                logits = raw / tau
            logits = torch.clamp(logits, min=-50.0, max=50.0)
            probs  = torch.softmax(logits, dim=1)
            probs  = self._apply_topk(probs)
        y = (feats * probs.to(feats.dtype)).sum(dim=1)
        y = self.post(y)
        return self.act(self.gn(y))

# ==============================
# Module A: AD-BiFPN_V2
# ==============================
try:
    from mmcv.ops import CARAFEPack
    _HAS_CARAFE = True
except Exception:
    _HAS_CARAFE = False
    CARAFEPack = None

class Up2x(nn.Module):
    def __init__(self, c):
        super().__init__()
        if _HAS_CARAFE:
            self.op = CARAFEPack(channels=c, up_kernel=5, up_group=1,
                                  encoder_kernel=3, encoder_dilation=1, scale_factor=2)
        else:
            self.op = None
    def forward(self, x, size=None):
        if _HAS_CARAFE:
            with torch.cuda.amp.autocast(enabled=False):
                y = self.op(x.float())
            y = y.to(dtype=x.dtype)
            if size is not None and y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
            return y
        else:
            if size is not None:
                return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

class FastWeightedAdd(nn.Module):
    def __init__(self, n_in=2, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_in)); self.eps = eps
    def forward(self, xs):
        w = torch.relu(self.w); w = w / (torch.sum(w) + self.eps)
        out = 0
        for i, x in enumerate(xs): out = out + w[i] * x
        return out

class ADBiFPN_V2(nn.Module):
    def __init__(self, c, gn_groups=8):
        super().__init__()
        self.lat2 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), GN(c, gn_groups), nn.SiLU(True))
        self.lat3 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), GN(c, gn_groups), nn.SiLU(True))
        self.lat4 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), GN(c, gn_groups), nn.SiLU(True))
        self.lat5 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), GN(c, gn_groups), nn.SiLU(True))
        self.w4_td = FastWeightedAdd(2); self.conv4_td = DeformableConvBlockV2(c, c, with_offset=True)
        self.w3_td = FastWeightedAdd(2); self.conv3_td = DeformableConvBlockV2(c, c, with_offset=True)
        self.w2_td = FastWeightedAdd(2); self.conv2_td = DeformableConvBlockV2(c, c, with_offset=True)
        self.w3_bu = FastWeightedAdd(2); self.conv3_bu = DeformableConvBlockV2(c, c, with_offset=True)
        self.w4_bu = FastWeightedAdd(2); self.conv4_bu = DeformableConvBlockV2(c, c, with_offset=True)
        self.up2 = Up2x(c)
        self.guide2 = nn.Sequential(nn.Conv2d(c+1, c, 1, bias=False), GN(c), nn.SiLU(True))
        self.guide3 = nn.Sequential(nn.Conv2d(c+1, c, 1, bias=False), GN(c), nn.SiLU(True))
        self.guide4 = nn.Sequential(nn.Conv2d(c+1, c, 1, bias=False), GN(c), nn.SiLU(True))
        self.guide5 = nn.Sequential(nn.Conv2d(c+1, c, 1, bias=False), GN(c), nn.SiLU(True))
    @staticmethod
    def down(x, size): return F.adaptive_avg_pool2d(x, output_size=size)
    def forward(self, p2, p3, p4, p5, b2=None, b3=None, b4=None, b5=None):
        zeros = lambda t: torch.zeros(t.size(0), 1, t.size(2), t.size(3), device=t.device, dtype=t.dtype)
        b2 = b2 if b2 is not None else zeros(p2)
        b3 = b3 if b3 is not None else zeros(p3)
        b4 = b4 if b4 is not None else zeros(p4)
        b5 = b5 if b5 is not None else zeros(p5)
        p2 = self.lat2(p2); p3 = self.lat3(p3); p4 = self.lat4(p4); p5 = self.lat5(p5)
        p2g = self.guide2(torch.cat([p2, b2], dim=1))
        p3g = self.guide3(torch.cat([p3, b3], dim=1))
        p4g = self.guide4(torch.cat([p4, b4], dim=1))
        p5g = self.guide5(torch.cat([p5, b5], dim=1))
        p4_td_in = self.w4_td([p4g, self.up2(p5g, size=p4g.shape[-2:])]); p4_td = self.conv4_td(p4_td_in, guide=b4)
        p3_td_in = self.w3_td([p3g, self.up2(p4_td, size=p3g.shape[-2:])]); p3_td = self.conv3_td(p3_td_in, guide=b3)
        p2_td_in = self.w2_td([p2g, self.up2(p3_td, size=p2g.shape[-2:])]); p2_td = self.conv2_td(p2_td_in, guide=b2)
        p3_out_in = self.w3_bu([p3_td, self.down(p2_td, p3_td.shape[-2:])]); p3_out = self.conv3_bu(p3_out_in, guide=b3)
        p4_out_in = self.w4_bu([p4_td, self.down(p3_out, p4_td.shape[-2:])]); p4_out = self.conv4_bu(p4_out_in, guide=b4)
        p5_out = p5g
        return p2_td, p3_out, p4_out, p5_out

# ==============================
# Module B baseline: CBAM + ASPP refine
# ==============================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super().__init__()
        self.ChannelGate = ChannelAttention(in_channel, reduction_ratio)
        self.SpatialGate = SpatialAttention()
    def forward(self, x):
        ch = self.ChannelGate(x); x = ch * x
        sp = self.SpatialGate(x); x = sp * x
        return x

class MFF_CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super().__init__()
        self.cbam = CBAM(in_channel, reduction_ratio)
    def forward(self, feat, near_fg=None, near_bg=None):
        return self.cbam(feat)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                   GN(out_channels), nn.ReLU(inplace=True)]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False),
                         GN(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        size = x.shape[-2:]
        for m in self: x = m(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=48):
        super().__init__()
        mods = [nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), GN(out_channels), nn.ReLU(inplace=True))]
        for r in atrous_rates: mods.append(ASPPConv(in_channels, out_channels, r))
        mods.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(mods)
        self.project = nn.Sequential(nn.Conv2d(len(self.convs)*out_channels, out_channels, 1, bias=False),
                                     GN(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        res = [m(x) for m in self.convs]
        return self.project(torch.cat(res, dim=1))

# ==============================
# Other shared blocks
# ==============================
class GGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False, gn_groups=8):
        super().__init__()
        self.gate_conv = nn.Sequential(GN(in_channels+1, gn_groups),
                                       nn.Conv2d(in_channels+1, in_channels+1, 1, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels+1, 1, 1, bias=False),
                                       nn.Sigmoid())
        self.out_cov = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
    def forward(self, in_feat, gate_feat):
        att = self.gate_conv(torch.cat([in_feat, gate_feat], dim=1))
        return self.out_cov(in_feat * (att + 1))

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                                     nn.Sigmoid())
    def forward(self, x):
        y = self.avg_pool(x); y = self.conv_du(y); return x * y

class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act_cls=nn.PReLU):
        super().__init__()
        self.body = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act_cls(),
                                  conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)
    def forward(self, x):
        res = self.CA(self.body(x)); return x + res

class RFD(nn.Module):
    def __init__(self, channel, kernel_size, reduction, bias, act_cls=nn.PReLU, n_resblocks=2):
        super().__init__()
        blocks = [RCAB(channel, kernel_size, reduction, bias=bias, act_cls=act_cls) for _ in range(n_resblocks)]
        blocks.append(conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*blocks)
    def forward(self, x): return x + self.body(x)

# ==============================
# DABNet (A-only advanced, B baseline)
# ==============================
class DABNet(nn.Module):
    def __init__(self, channel=48, kernel_size=3, reduction=4, bias=False, act_cls=nn.PReLU,
                 n_resblocks=2, iteration=3, pvt_ckpt_path: str = './pvt_v2_b2.pth',
                 dasm_topk=2, dasm_use_gumbel=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        try:
            sd = torch.load(pvt_ckpt_path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            model_dict.update({k:v for k,v in sd.items() if k in model_dict})
            self.backbone.load_state_dict(model_dict, strict=False)
            print(f"✅ Loaded PVTv2-B2 from {pvt_ckpt_path}")
        except Exception as e:
            print(f"⚠️ PVTv2-B2 weights not loaded: {e}")
        self.iteration = iteration

        # A: per-level DASM (with C toggles)
        self.ctx_4 = DASM_V2(64,  channel, topk=dasm_topk, use_gumbel=dasm_use_gumbel)
        self.ctx_3 = DASM_V2(128, channel, topk=dasm_topk, use_gumbel=dasm_use_gumbel)
        self.ctx_2 = DASM_V2(320, channel, topk=dasm_topk, use_gumbel=dasm_use_gumbel)
        self.ctx_1 = DASM_V2(512, channel, topk=dasm_topk, use_gumbel=dasm_use_gumbel)

        # initial coarse (H/32)
        self.coarse_init_head = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                              nn.Conv2d(channel, 1, 1))

        # multi-scale edges (for AD-BiFPN guidance)
        self.ms_edges = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1, bias=False), GN(channel), nn.SiLU(True),
                          nn.Conv2d(channel, channel, 3, padding=1, bias=False), GN(channel), nn.SiLU(True),
                          nn.Conv2d(channel, 1, 1, bias=True)) for _ in range(4)
        ])

        # A: AD-BiFPN_V2
        self.ad_bifpn = ADBiFPN_V2(channel)

        # B baseline: MFF with CBAM
        self.mff_4 = MFF_CBAM(channel)
        self.mff_3 = MFF_CBAM(channel)
        self.mff_2 = MFF_CBAM(channel)
        self.mff_1 = MFF_CBAM(channel)

        # Gates & decoders
        self.gate_1 = GGA(channel, channel); self.gate_2 = GGA(channel, channel); self.gate_3 = GGA(channel, channel)
        self.rfd_1 = RFD(channel, kernel_size, reduction, bias, act_cls=act_cls, n_resblocks=n_resblocks)
        self.rfd_2 = RFD(2*channel, kernel_size, reduction, bias, act_cls=act_cls, n_resblocks=n_resblocks)
        self.rfd_3 = RFD(3*channel, kernel_size, reduction, bias, act_cls=act_cls, n_resblocks=n_resblocks)
        self.gate_conv   = BasicConv2d(channel, 1, 1)
        self.gate_conv_1 = BasicConv2d(channel, 1, 1)
        self.gate_conv_2 = BasicConv2d(2*channel, 1, 1)
        self.upsample_2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out  = BasicConv2d(3*channel, channel, 3, padding=1)
        self.pred = nn.Conv2d(channel, 1, 1)

        # B baseline: ASPP refine head
        self.refine_aspp = ASPP(2*channel, out_channels=channel)
        self.out_pred    = nn.Conv2d(channel, 1, 1)

    def _ms_edges(self, x4,x3,x2,x1):
        h2 = torch.sigmoid(self.ms_edges[0](x4))
        h3 = torch.sigmoid(self.ms_edges[1](x3))
        h4 = torch.sigmoid(self.ms_edges[2](x2))
        h5 = torch.sigmoid(self.ms_edges[3](x1))
        return h2,h3,h4,h5

    @staticmethod
    def build_prompt_uncert(prob, k=3, pow_alpha=1.5, eps=_LOG_EPS):
        with torch.cuda.amp.autocast(enabled=False):
            p = prob.float().clamp(min=_PROB_MIN, max=1.0 - _PROB_MIN)
            d = dilate01(p, k); e = erode01(p, k)
            near_fg = torch.clamp(d - p, 0, 1); near_bg = torch.clamp(p - e, 0, 1)
            ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            ent = ent / math.log(2.0); ent = ent.clamp(min=0).pow(pow_alpha)
        return near_fg.to(prob.dtype), near_bg.to(prob.dtype), ent.to(prob.dtype)

    def forward(self, x):
        H,W = x.size(2), x.size(3)
        feats = self.backbone(x)
        x4,x3,x2,x1 = feats[0],feats[1],feats[2],feats[3]

        # DASM per-level
        x4 = self.ctx_4(x4); x3 = self.ctx_3(x3); x2 = self.ctx_2(x2); x1 = self.ctx_1(x1)

        # initial coarse
        init_logits_low = self.coarse_init_head(x1)
        init_prob_low   = torch.sigmoid(init_logits_low)

        # edges + AD-BiFPN fusion
        b2,b3,b4,b5 = self._ms_edges(x4,x3,x2,x1)
        p2,p3,p4,p5 = self.ad_bifpn(x4,x3,x2,x1, b2=b2,b3=b3,b4=b4,b5=b5)
        x4_img,x3_img,x2_img,x1_img = p2,p3,p4,p5

        stage_pred, coarse_pred = [], None
        init_prob2 = F.interpolate(init_prob_low, size=x4_img.shape[-2:], mode='bilinear', align_corners=False)
        init_prob3 = F.interpolate(init_prob_low, size=x3_img.shape[-2:], mode='bilinear', align_corners=False)
        init_prob4 = F.interpolate(init_prob_low, size=x2_img.shape[-2:], mode='bilinear', align_corners=False)
        init_prob5 = init_prob_low
        x4d_for_refine = None

        for it in range(self.iteration):
            if coarse_pred is None:
                prob2,prob3,prob4,prob5 = init_prob2,init_prob3,init_prob4,init_prob5
            else:
                logits_low = self.pred(coarse_pred); prob_low = torch.sigmoid(logits_low)
                prob5 = prob_low
                prob4 = F.interpolate(prob_low, size=x2_img.shape[-2:], mode='bilinear', align_corners=False)
                prob3 = F.interpolate(prob_low, size=x3_img.shape[-2:], mode='bilinear', align_corners=False)
                prob2 = F.interpolate(prob_low, size=x4_img.shape[-2:], mode='bilinear', align_corners=False)

            # prompts (unused by CBAM, but kept for API consistency)
            nfg2, nbg2, _ = self.build_prompt_uncert(prob2, k=3)
            nfg3, nbg3, _ = self.build_prompt_uncert(prob3, k=3)
            nfg4, nbg4, _ = self.build_prompt_uncert(prob4, k=3)
            nfg5, nbg5, _ = self.build_prompt_uncert(prob5, k=3)

            x1d = self.mff_1(x1_img, nfg5, nbg5)
            if coarse_pred is not None:
                coarse_gate = F.interpolate(self.gate_conv(coarse_pred), size=x1d.shape[-2:], mode='bilinear', align_corners=False)
                x1d = self.gate_1(x1d, coarse_gate)
            x2_feed = self.rfd_1(x1d)

            x2d = self.mff_2(x2_img, nfg4, nbg4)
            if it > 0:
                x2_gate = self.upsample_2(self.gate_conv_1(x2_feed))
                x2d = self.gate_2(x2d, x2_gate)
            x3_feed = self.rfd_2(torch.cat((x2d, self.upsample_2(x2_feed)), dim=1))

            x3d = self.mff_3(x3_img, nfg3, nbg3)
            if it > 0:
                x3_gate = self.upsample_2(self.gate_conv_2(x3_feed))
                x3d = self.gate_3(x3d, x3_gate)

            x4d = self.mff_4(x4_img, nfg2, nbg2)
            x4_feed = self.rfd_3(torch.cat((x3d, self.upsample_2(x3_feed)), dim=1))
            coarse_pred = self.out(x4_feed)
            out_map     = self.pred(coarse_pred)
            pred_full   = F.interpolate(out_map, size=(H,W), mode='bilinear', align_corners=False)
            stage_pred.append(pred_full)
            x4d_for_refine = x4d

        # baseline ASPP refine
        x4_for_ref = F.adaptive_avg_pool2d(x4d_for_refine, output_size=coarse_pred.shape[-2:])
        fused      = torch.cat([coarse_pred, x4_for_ref], dim=1)
        refined    = self.refine_aspp(fused)
        logits_ref = self.out_pred(refined)
        final_pred = F.interpolate(logits_ref, size=(H,W), mode='bilinear', align_corners=False)
        return stage_pred, final_pred

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DABNet(channel=48, iteration=3).to(device)
    try:
        model = torch.compile(model)
        print("✅ torch.compile enabled")
    except Exception as e:
        print(f"⚠️ torch.compile not available: {e}")
    x = torch.randn(1,3,704,704, device=device)
    model.train()
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        s, f = model(x)
    print("stage preds:", [p.shape for p in s])
    print("final pred :", f.shape)
