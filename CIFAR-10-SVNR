# ============================================================
#                    Module imports
# ============================================================
# --- Standard libraries ---
import math
import os
import random

# --- Third-party ---
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from typing import Tuple

# --- PyTorch ecosystem ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchinfo import summary
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# ============================================================
#                    Device setting
# ============================================================
# Device - USES GPU IF AVAILABLE
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# ============================================================
#                      Hyperparameters
# ============================================================
# Network
batch_size = 128
lr = 1e-4
num_epochs = 200
warmup_steps = 5000

# Model
beta_start = 1e-7
beta_end = 0.02
T = 1000
lambda_val = 5
sigma_r_train = (1e-3, 5e-1)
sigma_s_train = (1e-4, 3e-1)
sigma_r_val = 0.1005
sigma_s_val = 0.02505

# Validation white level
w_val = 0.5

# Deterministic Data Loading
SEED = 12345
num_workers = 4
val_frac = 0.1  # validation fraction

# PSNR and SSIM Normalization Bounds
PSNR_MIN, PSNR_MAX = 10.0, 50.0  # dB   (clip for normalization)
SSIM_MIN, SSIM_MAX = 0.50, 1.00  # unit (clip for normalization)

# Pareto tolerance epsilons
EPS_SSIM = 7e-4  # 0.0007
EPS_PSNR = 0.3  # 0.3 dB


# ============================================================
# Image linearization classes/functions (SRGB <-> Linear)
# ============================================================
class SRGBToLinear(torch.nn.Module):
    """Inverse gamma sRGB -> linear RGB. Input: [0,1], Output: [0, ~1+]"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 0.055
        return torch.where(
            x <= 0.04045,
            x / 12.92,
            ((x + a) / (1 + a)) ** 2.4
        )


class InverseWhiteLevel(torch.nn.Module):
    """Divides by white level. In training: w ~ U[lo, hi]; in eval: w = val."""

    def __init__(self, mode: str, lo: float = 0.1, hi: float = 1.0, val: float = 0.5):
        super().__init__()
        assert mode in {"train", "eval"}
        self.mode, self.lo, self.hi, self.val = mode, lo, hi, val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (np.random.uniform(self.lo, self.hi) if self.mode == "train" else self.val)
        return x / float(w)


# Reprocessing for PSNR/SSIM computation and image visualization
def linear_to_srgb(x):
    a = 0.055
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * torch.pow(x, 1 / 2.4) - a)


def to_srgb_for_metrics(x_norm, w: float):
    # x_norm: [-1,1] (linear/w)  ->  sRGB [0,1]
    x_lin_divw = x_norm * 0.5 + 0.5  # [-1,1] -> [0,1]
    x_lin = torch.clamp(x_lin_divw * float(w), min=0.0)
    x_srgb = linear_to_srgb(x_lin)
    return x_srgb.clamp(0.0, 1.0)


# ============================================================
# CIFAR-10 - Deterministic Data loading with linearization
# ============================================================

# ------------------------------
# Seed
# ------------------------------
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED);
np.random.seed(SEED)
torch.manual_seed(SEED);
torch.cuda.manual_seed_all(SEED)


def seed_worker(worker_id: int):
    # Deterministic RNG for each worker
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------
# Separated Generators  (split/loader)
# ------------------------------------
g_split = torch.Generator().manual_seed(SEED)  # random_split
g_train = torch.Generator().manual_seed(SEED)  # DataLoader train (shuffle)
g_val = torch.Generator().manual_seed(SEED + 1)  # DataLoader val
g_test = torch.Generator().manual_seed(SEED + 2)  # DataLoader test

# ------------------------------
# Loader parameters
# ------------------------------
pin_memory = False
if torch.cuda.is_available():
    pin_memory = True

# ------------------------------
# Transforms (in [-1, 1])
# ------------------------------
train_transforms = transforms.Compose([
    transforms.ToTensor(),  # PIL -> [0,1]
    SRGBToLinear(),  # inverse gamma (sRGB -> linear)
    InverseWhiteLevel("train", 0.1, 1.0),  # w ~ U[0.1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    SRGBToLinear(),
    InverseWhiteLevel("eval", val=w_val),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_noisy_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ------------------------------
# Datasets
# ------------------------------
base_ds = datasets.CIFAR10(root='data',
                           train=True, download=True, transform=None
                           )
test_clean_ds = datasets.CIFAR10(root='data',
                                 train=False, download=True, transform=eval_transforms
                                 )
# optional
'''
test_noisy_ds = datasets.CIFAR10(root='data',
                                 train=False, download=True, transform=test_noisy_transforms
                                 )
'''
# ------------------------------
# Deterministic split train/val
# ------------------------------
n_val = int(len(base_ds) * val_frac)
n_train = len(base_ds) - n_val
train_raw, val_raw = random_split(base_ds, [n_train, n_val], generator=g_split)


# -------------------------------------------------------------------
# Application of different transforms to training and validation set
# -------------------------------------------------------------------
class WithTransform(Dataset):
    def __init__(self, subset: Dataset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        img, y = self.subset[idx]
        img = self.transform(img)
        return img, y


train_ds = WithTransform(train_raw, train_transforms)
val_ds = WithTransform(val_raw, eval_transforms)

# --------------------------------------
# DataLoaders (generator + seed_worker)
# --------------------------------------
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,  # dterministic shuffle by generator
    drop_last=True,  # avoids last "oscillatory" batch
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g_train,
    persistent_workers=(num_workers > 0),
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,  # fixed order
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g_val,
    persistent_workers=(num_workers > 0),
)

test_clean_loader = DataLoader(
    test_clean_ds,
    batch_size=batch_size,
    shuffle=False,  # fixed order
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g_test,
    persistent_workers=(num_workers > 0),
)

print(f"Ready (CIFAR-10): train={len(train_ds)}, val={len(val_ds)}, test={len(test_clean_ds)}")


# ============================================================
#                        Network
# ============================================================

# --------- Helper: sinusoidal embedding for scalar t ----------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B,1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -4, half, device=device)) * 2.0 * torch.pi
        # broadcasting ok: [B,1] * [half] -> [B,half]
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # padding in case of odd dimensions
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]


# --------- FiLM on feature map from condition vector ----------
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x, cond_vec):
        # x: [B,C,H,W], cond_vec: [B,cond_dim]
        scale, shift = self.to_scale_shift(cond_vec).chunk(2, dim=1)  # [B,C], [B,C]
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift


# --------- Residual Block with AdaGN+FiLM (time) + spatial tmap  ----------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout_p=0.0, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch, eps=1e-6)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch, eps=1e-6)
        self.dropout = nn.Dropout2d(dropout_p)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        # FiLM from time embedding (vector -> scale/shift per channel)
        self.film = FiLM(time_dim, out_ch)

        # conditional projection from spatial tmap (optional), map -> channels out_ch
        self.tmap_proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)

        self.skip = (in_ch != out_ch)
        if self.skip:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, temb_vec, tmap_feat=None):
        h = self.conv1(self.act(self.norm1(x)))
        # injection from time (FiLM on h)
        h = self.film(h, temb_vec)

        # injection from spatial tmap (if present): additive after FiLM
        if tmap_feat is not None:
            # tmap_feat: [B,out_ch,H,W]
            h = h + self.tmap_proj(tmap_feat)

        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        if self.skip:
            x = self.skip_conv(x)
        return x + h


# --------- Attention 2D (MHSA) ----------
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, self.num_heads, c // self.num_heads, h * w)  # [B,H,C/H,HW]
        k = self.k(x_norm).reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v(x_norm).reshape(b, self.num_heads, c // self.num_heads, h * w)

        attn = torch.einsum('bhct,bhcs->bhts', q, k) * (c // self.num_heads) ** -0.5  # [B,H,HW,HW]
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhts,bhcs->bhct', attn, v).reshape(b, c, h, w)
        out = self.proj(out)
        return x + out


# --------- Down/Up ----------
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.tr = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.tr(x)


# --------- tmap Encoder (spatial) ----------
class TMapEncoder(nn.Module):
    """
    Takes tmap in [B,1,32,32] and produces features to the 4 scales:
    32x32 -> C
    16x16 -> 2C
    8x8   -> 4C
    4x4   -> 4C
    """

    def __init__(self, base_ch=64):
        super().__init__()
        C = base_ch
        self.s32 = nn.Sequential(
            nn.Conv2d(1, C, 3, 1, 1), nn.GELU(),
            nn.Conv2d(C, C, 3, 1, 1), nn.GELU()
        )
        self.s16 = nn.Sequential(
            nn.Conv2d(C, 2 * C, 4, 2, 1), nn.GELU(),
            nn.Conv2d(2 * C, 2 * C, 3, 1, 1), nn.GELU()
        )
        self.s8 = nn.Sequential(
            nn.Conv2d(2 * C, 4 * C, 4, 2, 1), nn.GELU(),
            nn.Conv2d(4 * C, 4 * C, 3, 1, 1), nn.GELU()
        )
        self.s4 = nn.Sequential(
            nn.Conv2d(4 * C, 4 * C, 4, 2, 1), nn.GELU(),
            nn.Conv2d(4 * C, 4 * C, 3, 1, 1), nn.GELU()
        )

    def forward(self, tmap):
        f32 = self.s32(tmap)  # [B, C, 32,32]
        f16 = self.s16(f32)  # [B,2C, 16,16]
        f8 = self.s8(f16)  # [B,4C,  8, 8]
        f4 = self.s4(f8)  # [B,4C,  4, 4]
        return f32, f16, f8, f4


# ----------------------------------------+--
#    SVNR Enhanced U-Net (CIFAR 32x32)
# -------------------------------------------
class UNetSVNRPlus(nn.Module):
    def __init__(self, base_ch=64, time_dim=128, dropout_p=0.1,
                 attn_res=(16, 8), num_heads=4):
        super().__init__()
        C = base_ch
        self.time_dim = time_dim
        self.attn_res = set(attn_res)

        # ---- time embedding: scalar t (tmap average) -> sinusoid -> MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

        # ---- tmap spatial encoder
        self.tenc = TMapEncoder(base_ch=C)

        # ---- Encoder path (32->16->8->4)
        self.inc = ResBlock(in_ch=6, out_ch=C, time_dim=time_dim, dropout_p=dropout_p)  # 32x32
        self.attn32 = AttentionBlock(C) if 32 in self.attn_res else nn.Identity()

        self.down1 = Downsample(C, 2 * C)  # 16x16
        self.enc1 = ResBlock(2 * C, 2 * C, time_dim, dropout_p)
        self.attn16 = AttentionBlock(2 * C, num_heads) if 16 in self.attn_res else nn.Identity()

        self.down2 = Downsample(2 * C, 4 * C)  # 8x8
        self.enc2 = ResBlock(4 * C, 4 * C, time_dim, dropout_p)
        self.attn8 = AttentionBlock(4 * C, num_heads) if 8 in self.attn_res else nn.Identity()

        self.down3 = Downsample(4 * C, 4 * C)  # 4x4
        self.enc3 = ResBlock(4 * C, 4 * C, time_dim, dropout_p)

        # ---- Bottleneck
        self.bot1 = ResBlock(4 * C, 4 * C, time_dim, dropout_p)
        self.bot2 = ResBlock(4 * C, 4 * C, time_dim, dropout_p)

        # ---- Decoder path (4->8->16->32) with skip
        self.up1 = Upsample(4 * C, 2 * C)  # 8x8
        self.dec1 = ResBlock(2 * C + 4 * C, 2 * C, time_dim, dropout_p)  # concat with enc2
        self.attn8d = AttentionBlock(2 * C, num_heads) if 8 in self.attn_res else nn.Identity()

        self.up2 = Upsample(2 * C, C)  # 16x16
        self.dec2 = ResBlock(C + 2 * C, C, time_dim, dropout_p)  # concat with enc1
        self.attn16d = AttentionBlock(C, num_heads) if 16 in self.attn_res else nn.Identity()

        self.up3 = Upsample(C, C)  # 32x32
        self.dec3 = ResBlock(C + C, C, time_dim, dropout_p)  # concat with inc

        self.out = nn.Conv2d(C, 3, kernel_size=1)

    def forward(self, y, x, tmap):
        """
        y:   [B,3,32,32]   (noisy)
        x:   [B,3,32,32]   (current estimate)
        tmap:[B,1,32,32]   (time map SVNR)
        """
        B, _, H, W = x.shape
        assert H == 32 and W == 32, "This architecture is thought for 32x32."

        # 1) time embedding (scalar) from tmap average
        t_scalar = tmap.mean(dim=[2, 3])  # [B,1]
        temb = self.time_mlp(t_scalar)  # [B, time_dim]

        # 2) spatial features from tmap to 4 levels
        t32, t16, t8, t4 = self.tenc(tmap)  # [B,C,32,32], [B,2C,16,16], [B,4C,8,8], [B,4C,4,4]

        # ---- Encoder
        x0 = torch.cat([y, x], dim=1)  # [B,6,32,32]
        e0 = self.inc(x0, temb, t32)  # [B,C,32,32]
        e0 = self.attn32(e0)

        h = self.down1(e0)  # [B,2C,16,16]
        e1 = self.enc1(h, temb, t16)  # [B,2C,16,16]
        e1 = self.attn16(e1)

        h = self.down2(e1)  # [B,4C,8,8]
        e2 = self.enc2(h, temb, t8)  # [B,4C,8,8]
        e2 = self.attn8(e2)

        h = self.down3(e2)  # [B,4C,4,4]
        e3 = self.enc3(h, temb, t4)  # [B,4C,4,4]

        # ---- Bottleneck
        b = self.bot1(e3, temb, None)
        b = self.bot2(b, temb, None)

        # ---- Decoder
        u = self.up1(b)  # [B,2C,8,8]
        u = torch.cat([u, e2], dim=1)  # [B,6C,8,8]
        u = self.dec1(u, temb, None)  # [B,2C,8,8]
        u = self.attn8d(u)

        u = self.up2(u)  # [B,C,16,16]
        u = torch.cat([u, e1], dim=1)  # [B,3C,16,16]
        u = self.dec2(u, temb, None)  # [B,C,16,16]
        u = self.attn16d(u)

        u = self.up3(u)  # [B,C,32,32]
        u = torch.cat([u, e0], dim=1)  # [B,2C,32,32]
        u = self.dec3(u, temb, None)  # [B,C,32,32]

        return self.out(u)  # [B,3,32,32]


# ===========================================================================
#  Model selection based on metrics  (Pareto check + TOPSIS score)
# ===========================================================================
best_state = None  # {'epoch', 'psnr', 'ssim', 'score'}


def normalize_fixed(x, lo, hi):
    if hi <= lo: return 0.0
    return float(torch.clamp(torch.tensor((x - lo) / (hi - lo)), 0.0, 1.0).item())


def dominates_eps(a, b, eps_ssim=EPS_SSIM, eps_psnr=EPS_PSNR):
    # Not-worse ("negative" margin )
    not_worse = (a["ssim"] >= b["ssim"] - eps_ssim) and (a["psnr"] >= b["psnr"] - eps_psnr)
    # Better in at least one metric ("positive" margin )
    at_least_one_better = (a["ssim"] >= b["ssim"] + eps_ssim) or (a["psnr"] >= b["psnr"] + eps_psnr)
    return not_worse and at_least_one_better


def topsis_score(p_n, s_n, w_p=0.5, w_s=0.5):
    eps = 1e-12
    d_best = math.sqrt(w_p * (1.0 - p_n) ** 2 + w_s * (1.0 - s_n) ** 2)
    d_worst = math.sqrt(w_p * (p_n - 0.0) ** 2 + w_s * (s_n - 0.0) ** 2)
    # TOPSIS computation: weighted distance from ideal (1,1) and anti-ideal (0,0)
    topsis_score = d_worst / (d_worst + d_best + eps)  # + eps avoids division by 0 # in [0,1], higher = better
    return topsis_score


def maybe_update(psnr, ssim, psnr_lo, psnr_hi, ssim_lo, ssim_hi, epoch, w_p=0.5, w_s=0.5, tol=1e-4):
    """
    Updates best solution according to: (1) Pareto (dominance, with tolerance)
    (2) TOPSIS on fixed scales [PSNR_MIN,PSNR_MAX], [SSIM_MIN,SSIM_MAX].
    """
    global best_state

    cand = {'epoch': epoch, 'psnr': psnr, 'ssim': ssim}

    if best_state is None:
        # normalization in [0,1] of the metrics, to make them comparable
        p_n = normalize_fixed(psnr, psnr_lo, psnr_hi)
        s_n = normalize_fixed(ssim, ssim_lo, ssim_hi)
        cand['score'] = topsis_score(p_n, s_n, w_p, w_s)
        best_state = cand
        return True, best_state

    # 1) Pareto
    if dominates_eps(cand, best_state):
        p_n = normalize_fixed(psnr, psnr_lo, psnr_hi)
        s_n = normalize_fixed(ssim, ssim_lo, ssim_hi)
        cand['score'] = topsis_score(p_n, s_n, w_p, w_s)
        best_state = cand
        return True, best_state

    if dominates_eps(best_state, cand):
        return False, best_state

    # If no solution dominates the other:
    # 2) TOPSIS playoff

    p_n = normalize_fixed(psnr, psnr_lo, psnr_hi)
    s_n = normalize_fixed(ssim, ssim_lo, ssim_hi)
    cand['score'] = topsis_score(p_n, s_n, w_p, w_s)

    if cand['score'] > best_state['score'] + tol:
        best_state = cand
        return True, best_state
    return False, best_state


# ============================================================
# Gamma schedule construction and gamma <-> t functions
# ============================================================

# ------------------------------------------------------------
#   Schedule: gamma[t] = lambda_val * (1 - alpha_bar[t])
#   beta ~ linspace [beta_start, beta_end], T steps
# ------------------------------------------------------------
def build_gamma_schedule(T, beta_start, beta_end, lambda_val):
    beta = torch.linspace(beta_start, beta_end, T, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    gamma_table = lambda_val * (1.0 - alpha_bar)  # shape [T], increasing
    return gamma_table


# ------------------------------------------------------------
#   gamma -> t  (linear interpolation on gamma_table)
#   Returns t in [0, T-1] with same shape of gamma
# ------------------------------------------------------------
def gamma_to_t_torch(gamma, gamma_table):
    # gamma_table: [T] (monotonic increasing)
    T = gamma_table.shape[0]
    # clamping gamma on its range
    g = torch.clamp(gamma, min=gamma_table[0], max=gamma_table[-1])

    idx1 = torch.searchsorted(gamma_table, g)  # in [0, T]
    idx1 = idx1.clamp(max=T - 1)
    idx0 = (idx1 - 1).clamp(min=0)

    g0 = gamma_table[idx0]
    g1 = gamma_table[idx1]
    denom = (g1 - g0).clamp_min(1e-12)
    w = (g - g0) / denom  # linear weight

    t = idx0.to(g.dtype) * (1.0 - w) + idx1.to(g.dtype) * w
    # cleaning and final clamp
    t = torch.nan_to_num(t, nan=0.0, posinf=float(T - 1), neginf=0.0)
    t = t.clamp(0.0, float(T - 1))
    return t


# ------------------------------------------------------------
#  t -> gamma  (inverse linear interpolation on gamma_table)
# ------------------------------------------------------------
def t_to_gamma_torch(t, gamma_table):
    T = gamma_table.shape[0]
    tt = t.clamp(0.0, float(T - 1))

    t0 = tt.floor().long()
    t1 = (t0 + 1).clamp(max=T - 1)

    w = (tt - t0.to(tt.dtype))  # weight between t0 and t1
    g0 = gamma_table[t0]
    g1 = gamma_table[t1]
    gamma = g0 * (1.0 - w) + g1 * w
    return gamma


# ============================================================
#                  Forward SVNR function
# ============================================================

# -------------------------------------------------------------------------------
#    Sampler SVNR full-Torch
#    Returns: (noisy=y, xt, tmap, eps_tilde, sigma_p) or (xt, tmap) if test=True
# -------------------------------------------------------------------------------
def sample_svnr(imgs, sigma_r, sigma_s, gamma_schedule, test=False):
    """
    imgs  : tensor [B,1,H,W] in [-1,1]
    sigma_r / sigma_s : float, (min,max), or broadcastable tensor to [B,1,1,1]
    gamma_schedule: tensor [T] (from build_gamma_schedule)
    """
    B, _, H, W = imgs.shape
    dev, dtype = imgs.device, imgs.dtype
    gamma_table = gamma_schedule.to(dev, dtype)

    # helper - converts scalar / tuple / tensor in [B,1,1,1] on device
    def prep(val):
        if torch.is_tensor(val):
            val = val.to(dev, dtype)
            return val.view(-1, *[1] * (imgs.ndim - 1)) if val.numel() > 1 else \
                val.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            lo, hi = map(float, val)
            return torch.empty(B, 1, 1, 1, device=dev, dtype=dtype).uniform_(lo, hi)
        return torch.tensor(float(val), device=dev, dtype=dtype).view(1, 1, 1, 1).expand(B, 1, 1, 1)

    sigma_r = prep(sigma_r)
    sigma_s = prep(sigma_s)

    # 1) sigma_p (shot+read) and gamma_T
    imgs01 = (imgs + 1.0) * 0.5  # [-1,1] -> [0,1]
    if test:
        imgs01 = imgs01.clamp(0.0, 1.0)  # only for test

    sigma_p = torch.sqrt(sigma_r ** 2 + sigma_s ** 2 * imgs01).clamp(min=1e-3)
    gamma_T = sigma_p ** 2  # [B,1,H,W]

    # 2) building noisy image -> clamp min to 0
    eps_T = torch.randn_like(imgs)
    noisy01 = (imgs01 + sigma_p * eps_T).clamp(min=0.0)  # .clamp(0.,1.)
    noisy = noisy01 * 2.0 - 1.0  # back to [-1,1]

    # 3) t^_map: gamma_T -> t_map, then casual shift t0 and residual t^ (tmap)
    tmap_full = gamma_to_t_torch(gamma_T, gamma_table)  # t* per-pixel
    t0 = torch.rand((), device=dev, dtype=dtype) * tmap_full.max()  # scalar U(0, max t*)
    tmap = (tmap_full - t0).clamp_min(0.0)  # residual t^

    # 4) gamma_t from t residual t^
    gamma_t = t_to_gamma_torch(tmap, gamma_table)

    # 5) Noise correlation: eps_tilde from (Eq. 11)
    eps = torch.tensor(1e-5, device=dev, dtype=dtype)
    ratio = (gamma_t / (gamma_T + eps)).clamp(min=eps.item(), max=1.0 - eps.item())
    sqrt_ratio = torch.sqrt(ratio)
    sqrt_remain = torch.sqrt(1.0 - ratio)
    eps_t = torch.randn_like(imgs)
    eps_tilde = sqrt_ratio * eps_T + sqrt_remain * eps_t

    # 6) x_t
    xt01 = (imgs01 + torch.sqrt(gamma_t) * eps_tilde).clamp(min=0.0)  # .clamp(0.,1.)
    xt = xt01 * 2.0 - 1.0  # back to [-1,1]

    if test:
        return xt, tmap

    return noisy, xt, tmap, eps_tilde, sigma_p


# ============================================================
#                  Reverse SVNR function
# ============================================================
def reverse_svnr(
        noisy,
        x_init,
        tmap,  # float
        model,
        gamma_schedule
):
    """
    Executes reverse SVNR pixel-wise starting from (noisy, x_init, tmap).
    Returns denoised x with same shape of x_init.
    """
    x = x_init
    dev, dtype = x.device, x.dtype
    gamma_table = gamma_schedule.to(dev, dtype).contiguous()
    T = gamma_table.shape[0]

    # Work copies
    x_cur = x.clone()
    t_cur = tmap.clone().to(dtype)  # float

    with torch.no_grad():
        while (t_cur > 0).any():
            # Noise prediction
            with torch.amp.autocast(device_type=dev.type):
                eps_hat = model(noisy, x_cur, t_cur)

                # Indexing gamma to integer times
            t_idx = t_cur.floor().long().clamp(0, T - 1)
            g_t = gamma_table[t_idx]
            g_tm1 = gamma_table[(t_idx - 1).clamp_min(0)]
            eta_t = g_t - g_tm1  # eq. (3)

            # x0_hat
            x0_hat = x_cur - torch.sqrt(g_t) * eps_hat  # from Eq. (11)
            #  reverse: from Eq. (8)
            coef_x = (g_tm1 / g_t).nan_to_num(0.0)
            coef_cnd = (eta_t / g_t).nan_to_num(0.0)
            coef_noi = torch.sqrt((g_tm1 * eta_t / g_t).clamp_min(0.0)).nan_to_num(0.0)
            eps_t_1 = torch.randn_like(x_cur)
            x_prev = coef_x * x_cur + coef_cnd * x0_hat + coef_noi * eps_t_1  # Eq. (8)

            # Updating current x (x_(t-1))
            mask_stop = (t_idx <= 1)  # True where discrete time is 1 or 0 (chain end), False otherwise
            x_cur = torch.where(mask_stop, x0_hat,
                                x_prev)  # where mask_stop = True -> x0_hat, where mask_stop = False -> x_prev

            # Times decreasing
            t_cur = (t_cur - 1.0).clamp_min(0.0)

    x_out = x_cur

    return x_out


# ===================================================================
#       Function to force tmap to be single-channel -> [B,1,H,W]
# ===================================================================
@torch.no_grad()
def ensure_single_channel_tmap(tmap: torch.Tensor) -> torch.Tensor:
    """
    Accepts [B,H,W], [B,1,H,W] o [B,3,H,W] and always returns [B,1,H,W].
    If it has 3 channels returns the max (as in SVNR: max RGB -> T*).
    """
    tm = tmap
    if tm.dim() == 3:  # [B,H,W]
        tm = tm.unsqueeze(1)
    if tm.size(1) != 1:  # ex. 3 channels
        tm = tm.max(dim=1, keepdim=True).values
    return tm.contiguous()


# ============================================================
#              Loss functions (identity + edge)
# ============================================================

def charbonnier(x, eps=1e-3):
    # more robust than MSE/L1
    return torch.sqrt(x * x + eps * eps)


def gradient(img):
    # img: [B,C,H,W]
    dx = img[..., :, 1:] - img[..., :, :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    return dx, dy


@torch.no_grad()
def low_t_mask_from_tmap(t_map, q=0.20):
    """
    t_map: [B,1,H,W] or [B,H,W]
    Returns mask [B,1,H,W] with 1 where t is in the lowest quantile (<= q), per-image.
    """
    tm = t_map.float()
    if tm.ndim == 3:
        tm = tm.unsqueeze(1)
    B = tm.size(0)
    masks = []
    for b in range(B):
        thr = torch.quantile(tm[b].reshape(-1), q)
        masks.append((tm[b] <= thr).float())
    return torch.stack(masks, dim=0)  # [B,1,H,W]


def reconstruct_x0_from_eps(x_t, eps_hat, sigma_p):
    """
    VE-style coherent: x_t ~= x0 + sigma_t * eps  ->  x0_hat = x_t - sigma_t * eps_hat
    sigma_p: [B,1,H,W] or broadcastable (returned from sample_svnr).
    """
    return x_t - sigma_p * eps_hat


def identity_and_edge_losses(x_t, y, eps_hat, t_map, sigma_p,
                             lambda_id=0.05, lambda_grad=0.02,
                             q_low_t=0.20, eps_charb=1e-3):
    """
    x_t, y, eps_hat: [B,C,H,W]
    t_map: [B,1,H,W] o [B,H,W]
    sigma_p: [B,1,H,W] (or broadcastable)
    """
    #  x0_hat reconstruction
    x0_hat = reconstruct_x0_from_eps(x_t, eps_hat, sigma_p)

    # low-t mask per-image
    low_mask = low_t_mask_from_tmap(t_map, q=q_low_t)  # [B,1,H,W]
    if low_mask.size(1) == 1 and x0_hat.size(1) != 1:
        low_mask = low_mask.expand(-1, x0_hat.size(1), -1, -1)

    # 1) Identity: x0_hat ~= y where t is low
    L_id = (charbonnier(x0_hat - y, eps=eps_charb) * low_mask).mean()

    # 2) Edge-preserving: aligns gradients (avoids detail smoothing)
    dx_hat, dy_hat = gradient(x0_hat)
    dx_y, dy_y = gradient(y)
    L_edge = (charbonnier(dx_hat - dx_y, eps=eps_charb).mean() +
              charbonnier(dy_hat - dy_y, eps=eps_charb).mean())

    L_aux = lambda_id * L_id + lambda_grad * L_edge
    return L_aux, {'L_id': L_id.detach(), 'L_edge': L_edge.detach()}


# ============================================================
#                        TRAINING
# ============================================================

# ------------------------------
# Model instantiation
# ------------------------------
svnr_model = UNetSVNRPlus().to(device)

# ------------------------------
# Gamma schedule
# ------------------------------
gamma_schedule = build_gamma_schedule(T, beta_start, beta_end, lambda_val)

# --------------------------------------------------------------
# Optimizer, scheduler (with warm up) and scaler instantiation
# --------------------------------------------------------------
optimizer = AdamW(
    svnr_model.parameters(),
    lr=lr,
    betas=(0.95, 0.999),
    eps=1e-8,
    weight_decay=1e-4  # moderate L2 to stop overfitting
)  # uses decoupled weight decay

total_steps = num_epochs * len(train_loader)
base_lr = optimizer.param_groups[0]["lr"]

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.10, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=base_lr * 0.10),
    ],
    milestones=[warmup_steps]
)
scaler = GradScaler(enabled=(device.type == "cuda"))

# ------------------------------
# Metrics on GPU
# ------------------------------
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# ----------------------------------------------
#          Training + Validation Loop
# ----------------------------------------------
best_train_loss = float('inf')
train_losses = []

for epoch in range(num_epochs):
    desc = f"Epoch {epoch + 1}/{num_epochs}"
    pbar = tqdm(train_loader, desc=desc, leave=True)
    running_loss = 0.0

    # ----- Training -----
    svnr_model.train()
    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)  # [B,3,32,32]

        # --- Forward SVNR ---
        noisy, xt, tmap, eps_tilde, sigma_p = sample_svnr(
            imgs, sigma_r_train, sigma_s_train, gamma_schedule
        )
        tmap = ensure_single_channel_tmap(tmap.to(device, non_blocking=True))

        with torch.amp.autocast(device_type=device.type):
            pred = svnr_model(noisy, xt, tmap)
            loss_mse = F.mse_loss(pred, eps_tilde)

            # ----- Identity + Edge Loss only at low t -----
            lambda_id = 0.02
            lambda_grad = 0.01
            L_aux, logs_aux = identity_and_edge_losses(
                x_t=xt, y=noisy, eps_hat=pred, t_map=tmap, sigma_p=sigma_p,
                lambda_id=lambda_id, lambda_grad=lambda_grad,
                q_low_t=0.15, eps_charb=1e-3
            )

            loss = loss_mse + L_aux

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(svnr_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            L_id=f"{logs_aux['L_id'].item():.4f}",
            L_edge=f"{logs_aux['L_edge'].item():.4f}"
        )

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch + 1} - Train loss: {epoch_train_loss:.4f}")

    # Saves model with minimum training loss
    if epoch_train_loss < best_train_loss:
        best_train_loss = epoch_train_loss
        torch.save(svnr_model.state_dict(), "SVNR_CIFAR_unet_heavy_linearization_lowest.pth")

    # ----- Validation every 10 epochs -----
    if (epoch + 1) % 10 == 0:
        svnr_model.eval()
        psnr_metric.reset();
        ssim_metric.reset()

        with torch.no_grad():
            for imgs_v, _ in val_loader:
                imgs_v = imgs_v.to(device, non_blocking=True)

                # ----- Forward SVNR -----
                noisy_v, x_v, tmap_v, _, _ = sample_svnr(
                    imgs_v, sigma_r_val, sigma_s_val, gamma_schedule
                )
                tmap_v = ensure_single_channel_tmap(tmap_v.to(device, non_blocking=True))

                # ----- Reverse SVNR -----
                x = reverse_svnr(noisy_v, x_v, tmap_v, svnr_model, gamma_schedule)

                # image reprocessing from linear RGB to sRGB (originals)
                den_v = to_srgb_for_metrics(x, w=w_val)  # [-1,1] linear/w -> sRGB [0,1]
                gt_v = to_srgb_for_metrics(imgs_v, w=w_val)  # [-1,1] linear/w -> sRGB [0,1]

                psnr_metric.update(den_v, gt_v)
                ssim_metric.update(den_v, gt_v)

        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        print(f"Epoch {epoch + 1} - Val PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # ----- Model selection: Pareto + fixed scale TOPSIS -----
        save_flag, state = maybe_update(
            avg_psnr, avg_ssim, PSNR_MIN, PSNR_MAX, SSIM_MIN, SSIM_MAX, epoch + 1,
            w_p=0.5, w_s=0.5, tol=1e-4
        )
        if save_flag:
            torch.save(svnr_model.state_dict(), "SVNR_CIFAR_unet_heavy_linearization_best.pth")
            print(f"[{epoch + 1}] NEW BEST: PSNR={state['psnr']:.2f} dB, "
                  f"SSIM={state['ssim']:.4f}, TOPSIS={state['score']:.4f}")

print("Training completed on:", device)

# -------------------------------------------------
#         Average Training Loss per Epoch Plot
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training Loss per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Optional plot save
plt.savefig("loss.png")
print("Saved plot to loss.png")

# ============================================================
#                        INFERENCE
# ============================================================

# ----------------------------------------------
# Loading weights and model inizialization
# ----------------------------------------------
svnr_model.load_state_dict(torch.load("SVNR_CIFAR_unet_heavy_linearization_best.pth", map_location=device))
# SVNR_CIFAR_unet_heavy_linearization_lowest.pth for lowest loss
svnr_model.eval()

# ====================================================================
#       Multiple Random Images Evaluation - Synthetic Noise Test
#     Using the fixed sigma_r_val and sigma_s_val to corrupt images
# ====================================================================
num_samples = 3
indices = random.sample(range(len(test_clean_ds)), num_samples)

svnr_model.eval()
fig = plt.figure(figsize=(9, 2.6 * num_samples), constrained_layout=False)
gs = fig.add_gridspec(num_samples, 3, wspace=0.05, hspace=0.42)

with torch.no_grad():
    for i, idx in enumerate(indices):
        # ---- Load image [1,3,32,32] ----
        img, _ = test_clean_ds[idx]
        img = img.unsqueeze(0).to(device)

        # ---- SVNR forward / reverse ----
        noisy, xt, tmap, _, _ = sample_svnr(img, sigma_r_val, sigma_s_val, gamma_schedule)

        #  make sure tmap is [B,1,H,W]
        tmap = ensure_single_channel_tmap(tmap)

        denoised = reverse_svnr(noisy, xt, tmap, svnr_model, gamma_schedule)

        # ---- sRGB for metrics/visualization ----
        noisy_srgb = to_srgb_for_metrics(noisy,    w_val)
        orig_srgb  = to_srgb_for_metrics(img,      w_val)
        den_srgb   = to_srgb_for_metrics(denoised, w_val)

        # ---- detach/cpu and (H,W,C) ----
        noisy_np = noisy_srgb.detach().cpu().squeeze(0).permute(1,2,0).numpy()  # [32,32,3]
        orig_np  =  orig_srgb.detach().cpu().squeeze(0).permute(1,2,0).numpy()
        den_np   =   den_srgb.detach().cpu().squeeze(0).permute(1,2,0).numpy()

        # ---- Metrics ----
        psnr = peak_signal_noise_ratio(orig_np, den_np, data_range=1.0)
        ssim = structural_similarity(orig_np, den_np, data_range=1.0, channel_axis=2)
        t_min, t_max = round(float(tmap.min().item()), 2), round(float(tmap.max().item()), 2)

        # ---- Axes ----
        ax0 = fig.add_subplot(gs[i, 0])
        ax1 = fig.add_subplot(gs[i, 1])
        ax2 = fig.add_subplot(gs[i, 2])

        ax1.text(0.5, 1.16,
                 f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f} | Time map range: [{t_min}, {t_max}]",
                 transform=ax1.transAxes, ha='center', va='bottom', fontsize=11)

        ax0.set_title('Original', fontsize=10.5, pad=4)
        ax1.set_title('Noisy',    fontsize=10.5, pad=4)
        ax2.set_title('Denoised', fontsize=10.5, pad=4)


        ax0.imshow(orig_np)
        ax1.imshow(noisy_np)
        ax2.imshow(den_np)
        for a in (ax0, ax1, ax2):
            a.set_axis_off()

fig.subplots_adjust(top=0.98, bottom=0.03, left=0.02, right=0.98, hspace=0.45, wspace=0.05)
# Optional plot save
fig.savefig("cifar_3samples2.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.show()
plt.close(fig)

# =========================================================================
#            Model evaluation on entire test set (no visualization)
# =========================================================================
psnr_list = []
ssim_list = []

with torch.no_grad():
    for imgs, _ in test_clean_loader:
        imgs = imgs.to(device)

        # 1) Forward SVNR: sample
        noisy, xt, tmap, _, _ = sample_svnr(imgs, sigma_r_val, sigma_s_val, gamma_schedule)
        # 2) Reverse SVNR
        tmap = ensure_single_channel_tmap(tmap)
        denoised = reverse_svnr(noisy, xt, tmap, svnr_model, gamma_schedule)

        # 3) Image reprocessing from linear RGB to sRGB (original)
        orig_srgb = to_srgb_for_metrics(imgs, w_val)
        den_srgb = to_srgb_for_metrics(denoised, w_val)

        # 4) Batch-wise metrics computation
        orig_np = (orig_srgb.detach().cpu().permute(0, 2, 3,
                                                    1).numpy() + 1) / 2  # [B,C,H,W] -> [B,H,W,C], maps [-1,1] -> [0,1]
        den_np = (den_srgb.detach().cpu().permute(0, 2, 3, 1).numpy() + 1) / 2

        for o, d in zip(orig_np, den_np):
            psnr_list.append(peak_signal_noise_ratio(o, d, data_range=1.0))
            ssim_list.append(structural_similarity(o, d, data_range=1.0, channel_axis=2))

# Overall results
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)
print(f"Test set: Average PSNR: {mean_psnr:.2f}, Average SSIM: {mean_ssim:.4f} on {len(psnr_list)} images")

# ============================================================
#            PSNR and SSIM histograms on Test Set
# ============================================================
psnr_arr = np.array(psnr_list)
ssim_arr = np.array(ssim_list)

# --- Create single figure with two subplots side by side ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))  # width, height

# --- Left: PSNR ---
axes[0].hist(psnr_arr, bins=50, alpha=0.75)
axes[0].axvline(mean_psnr, color='k', linestyle='--', linewidth=1.5,
                label=f"Mean = {mean_psnr:.2f} dB")
axes[0].set_title("Histogram of PSNR on Test Set", fontsize=12)
axes[0].set_xlabel("PSNR (dB)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# --- Right: SSIM ---
axes[1].hist(ssim_arr, bins=50, alpha=0.75)
axes[1].axvline(mean_ssim, color='k', linestyle='--', linewidth=1.5,
                label=f"Mean = {mean_ssim:.3f}")
axes[1].set_title("Histogram of SSIM on Test Set", fontsize=12)
axes[1].set_xlabel("SSIM")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# --- Adjust spacing and show ---
fig.tight_layout(pad=2.0)
plt.show()

# --- Optional: figure save ---
fig.savefig("psnr_ssim_histograms_cifar.png", dpi=300, bbox_inches="tight")
print("Saved plot to psnr_ssim_histograms_cifar.png")

# ============================================================================================
#         Visualization of 20 samples from Test Set - Direct Denoising (no extra noise)
# ============================================================================================

# Random selection of 20 images from test set
num_samples = 20
dataset = test_noisy_loader.dataset
indices = np.random.choice(len(dataset), size=num_samples, replace=False)
samples = torch.stack([dataset[i][0] for i in indices]).to(device)  # [20,3,32,32] in [-1,1]

# Treat the images as observations y (no added noise)
y = samples.clone()  # [-1,1]

# Forward SVNR
xt, tmap = sample_svnr(y, sigma_r_val, sigma_s_val, gamma_schedule, test=True)
tmap = ensure_single_channel_tmap(tmap)

# Reverse SVNR starting from x=y
x = y.clone()
denoised = reverse_svnr(x, xt, tmap, svnr_model, gamma_schedule)

# Conversion for visualization: (B,H,W,C) in [0,1]
orig_np = (((samples.detach().cpu().permute(0, 2, 3, 1).numpy()) + 1) / 2).clip(0, 1)
den_np = (((denoised.detach().cpu().permute(0, 2, 3, 1).numpy()) + 1) / 2).clip(0, 1)

# Plot
fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 3))
for i in range(num_samples):
    axes[i, 0].imshow(orig_np[i], cmap='gray')
    axes[i, 0].set_title('Input (y)')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(den_np[i], cmap='gray')
    axes[i, 1].set_title('Denoised')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.show()

# Optional plot save
plt.savefig("Direct_denoising_20_samples.png")
print("Saved plot to Direct_denoising_20_samples.png")

# ============================================================================================
#     Visualization of 4 samples from Test Set (4x4 grid) - Direct Denoising
# ============================================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Random selection of 4 images from test set
num_samples = 4
dataset = test_clean_loader.dataset
indices = np.random.choice(len(dataset), size=num_samples, replace=False)
samples = torch.stack([dataset[i][0] for i in indices]).to(device)  # [B,3,H,W] in [-1,1]

# Treat the images as observations y (no added noise)
y = samples.clone()  # [-1,1]

# Forward SVNR
xt, tmap = sample_svnr(y, sigma_r_val, sigma_s_val, gamma_schedule, test=True)
tmap = ensure_single_channel_tmap(tmap)  # [B,1,H,W] o [B,H,W]

# Reverse SVNR starting from x=y
x = y.clone()
denoised = reverse_svnr(x, xt, tmap, svnr_model, gamma_schedule)

# ---- Linear -> sRGB for visualization ----
orig_srgb = to_srgb_for_metrics(samples,  w_val).clamp(0, 1)   # [B,3,H,W]
den_srgb  = to_srgb_for_metrics(denoised, w_val).clamp(0, 1)   # [B,3,H,W]

# ---- Conversion for matplotlib: (B,H,W,C) in [0,1] ----
orig_np = orig_srgb.detach().cpu().permute(0, 2, 3, 1).numpy()
den_np  = den_srgb.detach().cpu().permute(0, 2, 3, 1).numpy()

# --- Figure ---
fig = plt.figure(figsize=(10.0, 6.0))
# Grid: [Input1, Denoised1, SPACER, Input2, Denoised2]
gs = gridspec.GridSpec(
    2, 5,
    width_ratios=[1, 1, 0.15, 1, 1],
    wspace=0.08,
    hspace=0.35
)

for i in range(num_samples):
    r = i // 2
    c_in  = 0 if (i % 2) == 0 else 3
    c_den = c_in + 1

    # Input (y)
    ax_in = fig.add_subplot(gs[r, c_in])
    ax_in.imshow(orig_np[i])
    ax_in.set_title('Input (y)', fontsize=10, pad=3)
    ax_in.set_axis_off()

    # Denoised
    ax_dn = fig.add_subplot(gs[r, c_den])
    ax_dn.imshow(den_np[i])
    ax_dn.set_title('Denoised', fontsize=10, pad=3)
    ax_dn.set_axis_off()

# Optional plot save
fig.subplots_adjust(top=0.95, bottom=0.06, left=0.05, right=0.97)
out_path = "CIFAR_direct_denoising_4examples.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Saved plot to {out_path}")

plt.show()
plt.close(fig)

