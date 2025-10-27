# ============================================================
#                    Module imports
# ============================================================
# --- Standard libraries ---
import math
import os
import glob
import random
import argparse
import h5py
import json

# --- Third-party ---
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from time import perf_counter
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from typing import List, Tuple
import sigpy as sp

# --- PyTorch ecosystem ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils import data
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ============================================================
#                      Hyperparameters 
# ============================================================
# Network
batch_size      = 32
lr              = 1e-4
num_epochs      = 100

# Model
beta_start = 1e-4
beta_end = 0.02
T               = 1000
lambda_val      = 5
sigma_train   = (1e-2, 1.6e-1)
sigma_val     = 8.3e-2


# Deterministic Data Loading
SEED = 12345
num_workers = 4
val_frac = 0.1    # validation fraction

# PSNR and SSIM Normalization Bounds
PSNR_MIN, PSNR_MAX = 10.0, 50.0     # dB   (clip for normalization)
SSIM_MIN, SSIM_MAX = 0.35, 1.00     # unit (clip for normalization)

# Pareto tolerance epsilons
EPS_SSIM = 7e-4   # ~0.0007
EPS_PSNR = 0.3    # ~0.3 dB

# ============================================================
# FastMRI Knee Single Coil — Optimized Data Loading 
# ============================================================

# ---------- Seeds ----------

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def seed_worker(worker_id: int):
    ws = SEED + worker_id
    np.random.seed(ws); random.seed(ws)

g_train = torch.Generator().manual_seed(SEED)
g_val   = torch.Generator().manual_seed(SEED + 1)
g_test  = torch.Generator().manual_seed(SEED + 2)

# ------------------------------------------------------------
#                        Utilities
# ------------------------------------------------------------
def _list_h5(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "*.h5"))) if root and os.path.isdir(root) else []

def _split_files_by_volume(files: List[str], ratios=(0.8, 0.1, 0.1), seed=SEED):
    """Split deterministico per file (.h5)."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios devono sommare a 1."
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    n_test  = n - n_train - n_val
    return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

def _save_list(paths: List[str], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths: f.write(p + "\n")

# ============================================================
#             OPTIMIZED VERSION
# ============================================================
def precompute_norm_stats(files: List[str], recon="rss", out_json="norm_stats.json",
                          samples_per_vol=6, ds=4, pct_low=0.5, pct_high=99.5):
    """
    Computes vmin/vmax per-volume (sampling some downsampled slices) and saves on JSON.
    Executes ONCE per for a stable file set (train+val+test).
    """
    stats = {}
    for p in files:
        with h5py.File(p, "r") as f:
            rk = f"reconstruction_{recon}"
            if rk not in f:
                rk = "reconstruction_rss" if recon=="esc" else "reconstruction_esc"
            vol = f[rk]
            n = vol.shape[0]
            idx = np.linspace(0, n-1, num=min(samples_per_vol, n), dtype=int)
            vals = []
            for i in idx:
                a = vol[i]
                a = a[::ds, ::ds] if ds and ds > 1 else a
                vals.append(a.reshape(-1))
            vals = np.concatenate(vals, axis=0).astype(np.float64, copy=False)
            vmin = np.percentile(vals, pct_low)
            vmax = np.percentile(vals, pct_high)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(vals.min())
                vmax = float(vals.max()) if float(vals.max()) > vmin else vmin + 1.0
            stats[p] = [float(vmin), float(vmax)]
    with open(out_json, "w") as f:
        json.dump(stats, f)
    return out_json

class FastMRIReconSliceDatasetOptimized(Dataset):
    def __init__(self, files: List[str], recon="rss",
                 stats_json="norm_stats.json", patch_size=None, center_crop=False):
        self.files = sorted(files)
        self.recon = recon
        self.patch_size = patch_size
        self.center_crop = center_crop

        with open(stats_json, "r") as f:
            self.stats = json.load(f)

        self.index: List[Tuple[int,int]] = []
        self.file_rk = {}
        for fi, p in enumerate(self.files):
            with h5py.File(p, "r") as f:
                rk = f"reconstruction_{recon}"
                if rk not in f:
                    rk = "reconstruction_rss" if recon=="esc" else "reconstruction_esc"
                n = f[rk].shape[0]
            self.file_rk[fi] = rk
            self.index.extend([(fi, si) for si in range(n)])

        self._fh = {}  

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_fh"] = {} 
        return s

    def __len__(self): return len(self.index)

    def _get_handle(self, fi):
        h = self._fh.get(fi)
        if h is None or not h.id.valid:
            h = h5py.File(self.files[fi], "r")
            self._fh[fi] = h
        return h

    def _apply_crop(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size is None:
            return x
        _, H, W = x.shape
        ph = pw = int(self.patch_size)
        if ph > H or pw > W:
            return x
        if self.center_crop:
            top = (H - ph)//2; left = (W - pw)//2
        else:
            top = np.random.randint(0, H - ph + 1); left = np.random.randint(0, W - pw + 1)
        return x[:, top:top+ph, left:left+pw]

    def __getitem__(self, i):
        fi, si = self.index[i]
        f = self._get_handle(fi)
        rk = self.file_rk[fi]
        arr = f[rk][si]  # numpy (H,W)
        vmin, vmax = self.stats[self.files[fi]]
        x = torch.from_numpy(arr).to(torch.float32).unsqueeze(0)  # [1,H,W]
        x.sub_(vmin).div_(vmax - vmin + 1e-8).clamp_(0.0, 1.0)
        x = x.sub_(0.5).div_(0.5)  # [-1,1]
        x = self._apply_crop(x)
        return x, 0

    def __del__(self):
        for h in self._fh.values():
            try:
                if h and h.id.valid: h.close()
            except:
                pass

def make_optimized_loaders(base_dir, recon="rss", ratios=(0.8,0.1,0.1),
                           patch_size=256, batch_size=32,
                           nw_train=4, nw_eval=2, seed=SEED,
                           stats_json="norm_stats.json",
                           precompute_if_missing=True):
    """
    OPTIMIZED pipeline end-to-end:
      1) creates .h5 lists (train+val) and splits per-volume
      2) precompute vmin/vmax per-volume
      3) optimized dataset + fast DataLoader 
    """
    train_dir = os.path.join(base_dir, "singlecoil_train")
    val_dir   = os.path.join(base_dir, "singlecoil_val")

    files_train = _list_h5(train_dir)
    files_val   = _list_h5(val_dir)
    if not files_train and not files_val:
        raise FileNotFoundError("No .h5 file found in train/val.")

    all_files = files_train + files_val
    train_files, val_files, test_files = _split_files_by_volume(all_files, ratios=ratios, seed=seed)

    # precomputes once
    if precompute_if_missing and (not os.path.isfile(stats_json)):
        precompute_norm_stats(train_files + val_files + test_files, recon=recon, out_json=stats_json)

    ds_train = FastMRIReconSliceDatasetOptimized(train_files, recon=recon,
                                                 stats_json=stats_json, patch_size=patch_size, center_crop=False)
    ds_val   = FastMRIReconSliceDatasetOptimized(val_files,   recon=recon,
                                                 stats_json=stats_json, patch_size=None)
    ds_test  = FastMRIReconSliceDatasetOptimized(test_files,  recon=recon,
                                                 stats_json=stats_json, patch_size=None)

    pin = (torch.cuda.is_available())
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=nw_train, pin_memory=pin, worker_init_fn=seed_worker,
                              generator=g_train, persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=nw_eval, pin_memory=pin, worker_init_fn=seed_worker,
                              generator=g_val, persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=nw_eval, pin_memory=pin, worker_init_fn=seed_worker,
                              generator=g_test, persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader, test_loader, (train_files, val_files, test_files)

# ---------------------------------------------------------
#                     Loaders creation
# ---------------------------------------------------------
train_loader, val_loader, test_loader, (train_files, val_files, test_files) = \
    make_optimized_loaders("fastmri/extracted", recon="esc",
                           ratios=(0.8,0.1,0.1), patch_size=256, # patch size indicates the reduced image dimensions: here 320x320 ->  256x256
                           batch_size=32, nw_train=4, nw_eval=2, 
                           stats_json="norm_stats.json")

# Creation of a corresponding data loader for test evaluation (comparison between ESC and RSS reconstruction)
_, _, test_loader_rss, _ = \
    make_optimized_loaders("fastmri/extracted", recon="rss",
                           ratios=(0.8,0.1,0.1), patch_size=256, # patch size indicates the reduced image dimensions: here 320x320 ->  256x256
                           batch_size=32, nw_train=4, nw_eval=2, 
                           stats_json="norm_stats.json")
print('Volumes:', len(train_files), len(val_files), len(test_files))
print('Slices:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

# ==================================================
#             Data pre-processing for SVNR
#====================================================

# -----------------------------------------------------------
#    Global sigma estimate from image background (Rayleigh)
# -----------------------------------------------------------
@torch.no_grad()
def estimate_sigma_global_esc(A01: torch.Tensor, q: float = 0.10, ds: int = 4,
                                    method: str = "median",
                                    clamp_min: float = 0.005, clamp_max: float = 0.08) -> torch.Tensor:
    """
    Input A01: [B,1,H,W] in [0,1]
    Output σ: [B,1,1,1]
    """
    x = A01
    if x.dim() == 3: x = x.unsqueeze(0)               # [1,1,H,W] -> [B,1,H,W]
    if ds and ds > 1:
        x = x[..., ::ds, ::ds]                        # strided downsample -> non-contiguous
    B = x.shape[0]
    xf = x.reshape(B, -1)                             # reshape

    n = xf.shape[1]
    k = max(1, int(q * n))
    vals, _ = torch.topk(xf, k, dim=1, largest=False, sorted=False)   # [B,k]

    if method == "mle":
        sigma = torch.sqrt(vals.pow(2).mean(dim=1) / 2.0 + 1e-12)
    else:
        med = vals.kthvalue(k // 2, dim=1).values
        sigma = med / math.sqrt(2.0 * math.log(2.0) + 1e-12)

    sigma = sigma.clamp_(clamp_min, clamp_max).view(B,1,1,1).to(A01.device)
    return sigma

# -------------------------------------------------------
#                VST Deterministic Transform
# -------------------------------------------------------
@torch.no_grad()
def rice_vst_forward(A01: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8):
    """
    z01 = sqrt( max(A^2/σ^2 - 1/2, 0) ) / sqrt( max(1/σ^2 - 1/2, 0) )
    A01: [B,1,H,W] in [0,1]
    sigma: [B,1,1,1]
    """
    s2 = sigma * sigma
    denom = torch.sqrt(torch.clamp(1.0 / s2 - 0.5, min=eps))          # [B,1,1,1]
    num   = torch.sqrt(torch.clamp(A01*A01 / s2 - 0.5, min=0.0))      # [B,1,H,W]
    z01 = num / denom                                                  # in [0,1]
    return z01.clamp_(0.0, 1.0)                          

@torch.no_grad()
def rice_vst_inverse(z01: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8):
    """
    A = σ * sqrt( ( z01 * sqrt(1/σ^2 - 1/2) )^2 + 1/2 )
    """
    s2 = sigma * sigma
    denom = torch.sqrt(torch.clamp(1.0 / s2 - 0.5, min=eps))           # [B,1,1,1]
    base  = z01 * denom
    return torch.sqrt(torch.clamp(base*base + 0.5, min=eps) * s2)

# ============================================================
#                        Network
# ============================================================

# ------------------------------------------------------------
#                     Utility layers
# ------------------------------------------------------------
def _gn_groups(c, max_groups=16):
    g = min(max_groups, c)
    while g > 1 and (c % g != 0):
        g -= 1
    return g


class FiLMvec(nn.Module):
    """FiLM from conditioning vector [B, D] on features [B, C, H, W]."""
    def __init__(self, d_in, c_out):
        super().__init__()
        self.to_scale_shift = nn.Linear(d_in, 2 * c_out)

    def forward(self, x, v):
        # x: [B,C,H,W], v: [B,D]
        s, b = self.to_scale_shift(v).chunk(2, dim=1)  # [B,C], [B,C]
        s = s[..., None, None]
        b = b[..., None, None]
        return x * (1 + s) + b

# Optional: FilM Map for spatially-variant time conditioning - here commented
'''
class FiLMmap(nn.Module):
    """FiLM from spatial embedding [B, Cemb, H, W] on features [B, C, H, W]."""
    def __init__(self, c_out, c_emb):
        super().__init__()
        self.to_gamma = nn.Conv2d(c_emb, c_out, 1)
        self.to_beta = nn.Conv2d(c_emb, c_out, 1)

    def forward(self, x, m):
        return x * (1 + self.to_gamma(m)) + self.to_beta(m)
'''

class SinusoidalPosEmb(nn.Module):
    """ Sinusoidal embedding for scalar t normalized in [0,1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B,1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -4, half, device=device)) * 2.0 * math.pi
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]


# ------------------------------------------------------------
#                    Attention blocks
# ------------------------------------------------------------
class AttentionBlock(nn.Module):
    """Simple MHSA with GroupNorm (only on maps <= 32x32)."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(_gn_groups(channels), channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_n = self.norm(x)
        q = self.q(x_n).reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = self.k(x_n).reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v(x_n).reshape(b, self.num_heads, c // self.num_heads, h * w)
        attn = torch.einsum('bhct,bhcs->bhts', q, k) * (c // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhts,bhcs->bhct', attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


# ------------------------------------------------------------
#                    t_map encoders
# ------------------------------------------------------------

# Optional: definition of a spatially-variant time injection encoder - here commented
'''
class PerPixelTimeEmbedding(nn.Module):
    """Per-pixel embedding of t_map (still uses per-pixel channel)."""
    def __init__(self, c_emb=64, max_freq=10.0, n_freqs=8):
        super().__init__()
        self.max_freq = max_freq
        self.n_freqs = n_freqs
        self.proj = nn.Sequential(
            nn.Conv2d(2 * n_freqs + 1, c_emb, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_emb, c_emb, 1),
        )

    def forward(self, t_map):  # [B,1,H,W]
        B, _, H, W = t_map.shape
        t_norm = t_map
        freqs = torch.logspace(
            0,
            math.log10(self.max_freq),
            self.n_freqs,
            device=t_map.device,
            dtype=t_map.dtype
        ).view(1, self.n_freqs, 1, 1)
        angles = 2 * math.pi * t_norm * freqs
        sin, cos = torch.sin(angles), torch.cos(angles)
        feats = torch.cat([t_norm, sin, cos], dim=1)
        return self.proj(feats)  # [B,c_emb,H,W]
'''

class TMapPyramid(nn.Module):
    """Piramid of features from t_map by additive injection for each scale."""
    def __init__(self, base_ch):
        super().__init__()
        C = base_ch
        self.l0 = nn.Sequential(
            nn.Conv2d(1, C, 3, 1, 1), nn.GELU(),
            nn.Conv2d(C, C, 3, 1, 1), nn.GELU()
        )  # H
        self.l1 = nn.Sequential(
            nn.Conv2d(C, 2 * C, 4, 2, 1), nn.GELU(),
            nn.Conv2d(2 * C, 2 * C, 3, 1, 1), nn.GELU()
        )  # H/2
        self.l2 = nn.Sequential(
            nn.Conv2d(2 * C, 4 * C, 4, 2, 1), nn.GELU(),
            nn.Conv2d(4 * C, 4 * C, 3, 1, 1), nn.GELU()
        )  # H/4
        self.l3 = nn.Sequential(
            nn.Conv2d(4 * C, 4 * C, 4, 2, 1), nn.GELU(),
            nn.Conv2d(4 * C, 4 * C, 3, 1, 1), nn.GELU()
        )  # H/8

    def forward(self, t_map):
        f0 = self.l0(t_map)
        f1 = self.l1(f0)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        return f0, f1, f2, f3


# ------------------------------------------------------------
#                       ResBlock+
# ------------------------------------------------------------
class ResBlockPlus(nn.Module):
    """
    ResBlock with:
      - GN + SiLU + Conv x2
      - FiLM from time vector(scalar t -> sinusoidal -> MLP)
      - additive injection from t_map features on same scale
    """
    def __init__(self, in_ch, out_ch, time_dim, tfeat_ch=None, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.film_vec = FiLMvec(time_dim, out_ch)
        self.tproj = nn.Conv2d(tfeat_ch, out_ch, 1) if tfeat_ch is not None else None

    def forward(self, x, t_vec, t_feat=None):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.film_vec(h, t_vec)
        if t_feat is not None and self.tproj is not None:
            h = h + self.tproj(t_feat)
        h = self.conv2(self.drop(self.act(self.norm2(h))))
        return h + self.skip(x)


# ------------------------------------------------------------
#                      Down / Up 
# ------------------------------------------------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, 4, 2, 1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ------------------------------------------------------------
#                    SVNR U-Net MRI Pro
# ------------------------------------------------------------
class SVNRUNetMRI_Pro(nn.Module):
    """
    x_t, y   : [B, Cx, H, W] (Cx=1 for 2D, or Cx=K for 2.5D with K slices)
    t_map    : [B, 1,  H, W]
    output   : eps_hat [B, 1, H, W]
    """
    def __init__(
        self,
        base_ch=64,
        time_dim=128,
        per_pixel_emb_ch=64,
        num_heads=4,
        attn_res=(64, 32),
        in_slices=1,
        dropout=0.0
    ):
        super().__init__()
        C = base_ch
        Cin = 2 * in_slices + 1  # x_t, y, t_map

        # --- time embedding ---
        self.t_scalar = SinusoidalPosEmb(time_dim)  # from average t -> vector
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
       # Optional: per-pixel time embedding 
       # self.t_perpix = PerPixelTimeEmbedding(c_emb=per_pixel_emb_ch)  #

        # --- t_map pyramid ---
        self.tenc = TMapPyramid(base_ch=C)

        # --- encoder (H → H/8) ---
        self.inc1 = ResBlockPlus(Cin, C, time_dim, tfeat_ch=C, dropout=dropout)
        self.inc2 = ResBlockPlus(C, C, time_dim, tfeat_ch=C, dropout=dropout)
        self.attnH = AttentionBlock(C, num_heads) if 'H' in attn_res else nn.Identity()
        self.down1 = Down(C, 2 * C)

        self.enc1 = ResBlockPlus(2 * C, 2 * C, time_dim, tfeat_ch=2 * C, dropout=dropout)  # H/2
        self.enc2 = ResBlockPlus(2 * C, 2 * C, time_dim, tfeat_ch=2 * C, dropout=dropout)
        self.attnH2 = AttentionBlock(2 * C, num_heads) if (isinstance(attn_res, (list, set, tuple)) and (128 in attn_res)) else nn.Identity()
        self.down2 = Down(2 * C, 4 * C)

        self.enc3 = ResBlockPlus(4 * C, 4 * C, time_dim, tfeat_ch=4 * C, dropout=dropout)  # H/4
        self.enc4 = ResBlockPlus(4 * C, 4 * C, time_dim, tfeat_ch=4 * C, dropout=dropout)
        self.attnH4 = AttentionBlock(4 * C, num_heads) if 64 in attn_res else nn.Identity()
        self.down3 = Down(4 * C, 4 * C)

        # --- bottleneck (H/8) ---
        self.bot1 = ResBlockPlus(4 * C, 4 * C, time_dim, tfeat_ch=4 * C, dropout=dropout)
        self.bot2 = ResBlockPlus(4 * C, 4 * C, time_dim, tfeat_ch=4 * C, dropout=dropout)
        self.attnH8 = AttentionBlock(4 * C, num_heads) if 32 in attn_res else nn.Identity()

        # --- decoder (H/8 → H) ---
        self.up1 = Up(4 * C, 2 * C)  # H/4
        self.dec1 = ResBlockPlus(2 * C + 4 * C, 2 * C, time_dim, tfeat_ch=4 * C, dropout=dropout)
        self.dec2 = ResBlockPlus(2*C,       2*C, time_dim, tfeat_ch=4*C, dropout=dropout) 
        self.attnD4 = AttentionBlock(2 * C, num_heads) if 64 in attn_res else nn.Identity()

        self.up2 = Up(2 * C, C)  # H/2
        self.dec3 = ResBlockPlus(C + 2 * C, C, time_dim, tfeat_ch=2 * C, dropout=dropout)
        self.dec4 = ResBlockPlus(C,         C,   time_dim, tfeat_ch=2*C, dropout=dropout)
        self.attnD2 = AttentionBlock(C, num_heads) if (isinstance(attn_res, (list, set, tuple)) and (128 in attn_res)) else nn.Identity()

        self.up3 = Up(C, C)  # H
        self.dec5 = ResBlockPlus(C + C, C, time_dim, tfeat_ch=C, dropout=dropout)

        self.out = nn.Conv2d(C, 1, 3, 1, 1)

    def forward(self, x_t, y, t_map):
        B, _, H, W = x_t.shape
        # 1) Input preparation
        #tpp = self.t_perpix(t_map)  # [B, cemb, H, W]  # optional -> derives from per-pixel time embedding
        # time vector from scalar t (uses average t as a proxy)
        t_mean = t_map.mean(dim=[2, 3])  # [B,1]
        t_vec = self.time_mlp(self.t_scalar(t_mean))  # [B, time_dim]

        # t_map pyramid
        tH, tH2, tH4, tH8 = self.tenc(t_map)

        # concat input
        x = torch.cat([x_t, y, t_map], dim=1)  # [B, 2*in_slices+1, H, W]

        # --- encoder ---
        h = self.inc1(x, t_vec, tH)
        h = self.inc2(h, t_vec, tH)
        h = self.attnH(h)
        h1 = h
        h = self.down1(h)  # H/2

        h = self.enc1(h, t_vec, tH2)
        h = self.enc2(h, t_vec, tH2)
        h = self.attnH2(h)
        h2 = h
        h = self.down2(h)  # H/4

        h = self.enc3(h, t_vec, tH4)
        h = self.enc4(h, t_vec, tH4)
        h = self.attnH4(h)
        h3 = h
        h = self.down3(h)  # H/8

        # --- bottleneck ---
        h = self.bot1(h, t_vec, tH8)
        h = self.bot2(h, t_vec, tH8)
        h = self.attnH8(h)

        # --- decoder ---
        h = self.up1(h)  # H/4
        h = torch.cat([h, h3], dim=1)
        h = self.dec1(h, t_vec, tH4)
        h = self.dec2(h, t_vec, tH4)  
        h = self.attnD4(h)

        h = self.up2(h)  # H/2
        h = torch.cat([h, h2], dim=1)
        h = self.dec3(h, t_vec, tH2)
        h = self.dec4(h, t_vec, tH2)
        h = self.attnD2(h)

        h = self.up3(h)  # H
        h = torch.cat([h, h1], dim=1)
        h = self.dec5(h, t_vec, tH)

        return self.out(h)

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
    at_least_one_better = (a["ssim"] >= b["ssim"] + eps_ssim) or  (a["psnr"] >= b["psnr"] + eps_psnr)
    return not_worse and at_least_one_better

def topsis_score(p_n, s_n, w_p=0.5, w_s=0.5):
    eps = 1e-12
    d_best  = math.sqrt(w_p*(1.0 - p_n)**2 + w_s*(1.0 - s_n)**2)
    d_worst = math.sqrt(w_p*(p_n - 0.0)**2 + w_s*(s_n - 0.0)**2)
    # TOPSIS computation: weighted distance from ideal (1,1) and anti-ideal (0,0)
    topsis_score   = d_worst / (d_worst + d_best + eps)  # + eps avoids division by 0 # ∈[0,1], higher = better
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
    gamma_table = lambda_val * (1.0 - alpha_bar)            # shape [T], increasing
    return gamma_table.to(device)

# ------------------------------------------------------------
#   gamma -> t  (linear interpolation on gamma_table)
#   Returns t in [0, T-1] with same shape of gamma
# ------------------------------------------------------------
def gamma_to_t_torch(gamma, gamma_table):
    # gamma_table: [T] (monotonic increasing)
    T = gamma_table.shape[0]
    # clamping gamma on its range
    g = torch.clamp(gamma, min=gamma_table[0], max=gamma_table[-1])

    idx1 = torch.searchsorted(gamma_table, g)               # in [0, T]
    idx1 = idx1.clamp(max=T-1)
    idx0 = (idx1 - 1).clamp(min=0)

    g0 = gamma_table[idx0]
    g1 = gamma_table[idx1]
    denom = (g1 - g0).clamp_min(1e-12)
    w = (g - g0) / denom                                    # linear weight

    t = idx0.to(g.dtype) * (1.0 - w) + idx1.to(g.dtype) * w
    # cleaning and final clamp 
    t = torch.nan_to_num(t, nan=0.0, posinf=float(T-1), neginf=0.0)
    t = t.clamp(0.0, float(T-1))
    return t


# ------------------------------------------------------------
#  t -> gamma  (inverse linear interpolation on gamma_table)
# ------------------------------------------------------------
def t_to_gamma_torch(t, gamma_table):
    T = gamma_table.shape[0]
    tt = t.clamp(0.0, float(T-1))

    t0 = tt.floor().long()
    t1 = (t0 + 1).clamp(max=T-1)

    w = (tt - t0.to(tt.dtype))                              # weight between t0 and t1
    g0 = gamma_table[t0]
    g1 = gamma_table[t1]
    gamma = g0 * (1.0 - w) + g1 * w
    return gamma

# ============================================================
#         Forward SVNR function with VST transformed input
# ============================================================

# -------------------------------------------------------------------------------
#    Input: the VST transformed z01, obtained by applying VST to A01
#    Output: (noisy=y, xt, tmap, eps_tilde, sigma_p) or (xt, tmap) if test=True
# -------------------------------------------------------------------------------
def sample_svnr_vst(z01, sigma, gamma_schedule, test=False):
    """
    z01  : tensor [B,1,H,W] in [0,1] - z01 obtained by VST
    sigma : float, (min,max), or broadcastable tensor to [B,1,1,1]
    gamma_schedule: tensor [T] (from build_gamma_schedule)
    """
    B, _, H, W = z01.shape
    dev, dtype = z01.device, z01.dtype
    gamma_table = gamma_schedule.to(dev, dtype)
    
    # helper – converts scalar / tuple / tensor in [B,1,1,1] on device
    def prep(val):
        if torch.is_tensor(val):
            val = val.to(dev, dtype)
            return val.view(-1, *[1]*(z01.ndim-1)) if val.numel() > 1 else \
                   val.view(1,1,1,1).expand(B,1,1,1)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            lo, hi = map(float, val)
            return torch.empty(B,1,1,1, device=dev, dtype=dtype).uniform_(math.log(lo), math.log(hi)).exp_()
        return torch.tensor(float(val), device=dev, dtype=dtype).view(1,1,1,1).expand(B,1,1,1)

    sigma = prep(sigma)
    
    if test:
        z01 = z01.clamp(0.0, 1.0)                    # only for test

    # 1) gamma_T
    sigma_p = torch.sqrt(sigma**2 * z01).clamp(min=1e-3)
    gamma_T = sigma_p**2                                   # [B,1,H,W]

    # 2) building noisy image –> clamp min to 0
    eps_T = torch.randn_like(z01)                          # [-1,1] -> [0,1]
    noisy01 = (z01 + sigma_p * eps_T).clamp(min=0.0)   #.clamp(0.,1.)
    noisy = noisy01 * 2.0 - 1.0                            # back to [-1,1]

    # 3) t̂_map: gamma_T -> t_map, then casual shift t0 and residual t̂ (tmap)
    tmap_full = gamma_to_t_torch(gamma_T, gamma_table)     # t* per-pixel
    t0 = torch.rand((), device=dev, dtype=dtype) * tmap_full.max()   # scalar U(0, max t*)
    tmap = (tmap_full - t0).clamp_min(0.0)                 # residual t̂
    
    # 4) gamma_t from t residual t̂
    gamma_t = t_to_gamma_torch(tmap, gamma_table)

    # 5) Noise correlation: eps_tilde from (Eq. 11)
    eps = torch.tensor(1e-5, device=dev, dtype=dtype)
    ratio = (gamma_t / (gamma_T + eps)).clamp(min=eps.item(), max=1.0 - eps.item())
    sqrt_ratio = torch.sqrt(ratio)
    sqrt_remain = torch.sqrt(1.0 - ratio)
    eps_t = torch.randn_like(z01)
    eps_tilde = sqrt_ratio * eps_T + sqrt_remain * eps_t

    # 6) x_t 
    xt01 = (z01 + torch.sqrt(gamma_t) * eps_tilde).clamp(min=0.0)   # .clamp(0.,1.)
    xt = xt01 * 2.0 - 1.0             # back to [-1,1]

    if test:
        return xt, tmap

    return noisy, xt, tmap, eps_tilde, sigma_p

# ============================================================
#                  Reverse SVNR function
# ============================================================
def reverse_svnr(
    noisy,           
    x_init,          
    tmap,          #  float
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
            g_t   = gamma_table[t_idx]
            g_tm1 = gamma_table[(t_idx - 1).clamp_min(0)]
            eta_t = g_t - g_tm1  # eq. (3)

            # x0_hat 
            x0_hat   = x_cur - torch.sqrt(g_t) * eps_hat # from Eq. (11)
            #  reverse: from Eq. (8)
            coef_x   = (g_tm1 / g_t).nan_to_num(0.0)
            coef_cnd = (eta_t / g_t).nan_to_num(0.0)
            coef_noi = torch.sqrt((g_tm1 * eta_t / g_t).clamp_min(0.0)).nan_to_num(0.0)
            eps_t_1 = torch.randn_like(x_cur)
            x_prev = coef_x * x_cur + coef_cnd * x0_hat + coef_noi * eps_t_1 # Eq. (8)
  

            # Updating current x (x_(t-1))
            mask_stop = (t_idx <= 1) #  True where discrete time is 1 or 0 (chain end), False otherwise
            x_cur = torch.where(mask_stop, x0_hat, x_prev) # where mask_stop = True -> x0_hat, where mask_stop = False -> x_prev

            # Times decreasing
            t_cur = (t_cur - 1.0).clamp_min(0.0)
    
    x_out = x_cur
    
    return x_out

# ============================================================
#                        TRAINING
# ============================================================

# ------------------------------
# Model instantiation
# ------------------------------
svnr_model = SVNRUNetMRI_Pro(
    base_ch=48,
    dropout=0.0
).to(device)  

# ------------------------------
# Gamma schedule
# ------------------------------
gamma_schedule = build_gamma_schedule(T,beta_start,beta_end,lambda_val)

# ----------------------------------------------
# Optimizer, scheduler and scaler instantiation
# ----------------------------------------------
optimizer = AdamW(
    svnr_model.parameters(),
    lr=lr,
    betas=(0.9, 0.95),   
    eps=1e-8,
    weight_decay=1e-4, # moderate L2 to stop overfitting
    fused = True 
)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
scaler = GradScaler()

# ------------------------------
# Metrics on GPU
# ------------------------------
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# ---------------------------------
#         Speed pack CUDA   
# ---------------------------------
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32 = True
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    svnr_model = svnr_model.to(device, memory_format=torch.channels_last)
    try:
        svnr_model = torch.compile(svnr_model)  # PyTorch 2.x
    except Exception as e:
        print("torch.compile skipped:", e)

# ------------------------------------------
#        Computational time count
# ------------------------------------------
def format_dhms(seconds: float) -> str:
    s = int(round(seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    parts = []
    if d: parts.append(f"{d} {'day' if d==1 else 'days'}")
    if h: parts.append(f"{h} {'hour' if h==1 else 'hours'}")
    if m: parts.append(f"{m} {'minute' if m==1 else 'minutes'}")
    if s or not parts: parts.append(f"{s} {'second' if s==1 else 'seconds'}")
    return " ".join(parts)

# ----------------------------------------------
#          Training + Validation Loop 
# ----------------------------------------------

best_train_loss = float('inf')
train_losses = []
total_start = perf_counter()  # [TIMER] start

for epoch in range(num_epochs):
    desc = f"Epoch {epoch+1}/{num_epochs}"
    pbar = tqdm(train_loader, desc=desc, leave=True)
    running_loss = 0.0

    # ----- Training -----
    svnr_model.train()
    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        # for faster computation
        if device.type == "cuda":
            imgs = imgs.to(memory_format=torch.channels_last)

        # ----- VST prep -----
        A01  = (imgs + 1.0) * 0.5    # [-1,1] -> [0,1]
        with torch.no_grad():
            sigma_data = estimate_sigma_global_esc(A01, q=0.10, ds=4)      # [B,1,1,1]
            z01 = rice_vst_forward(A01, sigma_data)                      # [B,1,H,W]
        
        # ----- Forward SVNR with VST modified input -----
        noisy, xt, tmap, eps_tilde,sigma_p = sample_svnr_vst(
            z01, sigma_train, gamma_schedule
        )
        # Image construction with Gaussian noise in [-1,1]
        z11=z01*2-1  # [0,1] -> [-1,1]
        # ----- Identity mix-in (no-noise) 5%: teaches to "do nothing" in t=0 -----
        #       xt = imgs, noisy = imgs, tmap = 0, target = 0
        if torch.rand(1, device=device).item() < 0.05:
            noisy = z11
            xt = z11
            tmap = torch.zeros_like(tmap)
            eps_tilde = torch.zeros_like(eps_tilde)

        # ----- Oversampling t0 = 0 with p = 1% -----
        if torch.rand(1, device=device).item() < 0.01:
            xt = noisy
            tmap = gamma_to_t_torch((sigma_p ** 2), gamma_schedule).clamp_min(0.0).to(dtype=torch.float32)
            eps_tilde = (noisy - z11) / (sigma_p + 1e-12)

        # for faster computation
        if device.type == "cuda":
            noisy = noisy.contiguous(memory_format=torch.channels_last)
            xt = xt.contiguous(memory_format=torch.channels_last)
            tmap = tmap.contiguous(memory_format=torch.channels_last)
            eps_tilde = eps_tilde.contiguous(memory_format=torch.channels_last)
       
        with torch.amp.autocast(device_type=device.type):
            pred = svnr_model(noisy, xt, tmap)
            loss_mse = F.mse_loss(pred, eps_tilde)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_mse).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(svnr_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss_mse.item() * imgs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch+1} — Train loss: {epoch_train_loss:.4f}")

    # Saves model with minimum training loss
    if epoch_train_loss < best_train_loss:
        best_train_loss = epoch_train_loss
        torch.save(svnr_model.state_dict(), "MRI_Rician_lowest.pth")

    scheduler.step()

    # ----- Validation every 10 epochs -----
    if (epoch + 1) % 10 == 0:
        svnr_model.eval()
        psnr_metric.reset()
        ssim_metric.reset()

        with torch.no_grad():
            for imgs_v, _ in val_loader:
                imgs_v = imgs_v.to(device, non_blocking=True)
                if device.type == "cuda":
                    imgs_v = imgs_v.contiguous(memory_format=torch.channels_last)
                    
                # ----- VST prep -----
                A01_v  = (imgs_v + 1.0) * 0.5
                with torch.no_grad():
                    sigma_data_v = estimate_sigma_global_esc(A01_v, q=0.10, ds=4)      # [B,1,1,1]
                    z01_v = rice_vst_forward(A01_v, sigma_data_v)                      # [B,1,H,W]
                    #print(z01_v.shape)

                # 1) generates ONLY noisy y (ignores xt/tmap from sampler)
                noisy_v, _, _, _,_ = sample_svnr_vst(z01_v, sigma_val, gamma_schedule)
                if device.type == "cuda":
                    noisy_v = noisy_v.contiguous(memory_format=torch.channels_last)

                # 2) builds T* from y: gamma_T* = σ_r^2 + σ_s^2 · clip(y,0,1)
                y01 = ((noisy_v + 1) / 2).clamp(0.0, 1.0)      # [0,1]
                gamma_Tstar = (sigma_val ** 2) * y01
                tmap_Tstar  = gamma_to_t_torch(gamma_Tstar, gamma_schedule).clamp_min(0.0)

                # 3) Reverse SVNR (Alg. 2)
                x_hat = reverse_svnr(
                    noisy_v,      # y (conditioning)
                    noisy_v,      # x_init = y
                    tmap_Tstar,   # T*
                    svnr_model,
                    gamma_schedule
                )

                # 4) metrics
                x_hat01 = ((x_hat + 1) / 2).clamp(0.0, 1.0)
                A_den=rice_vst_inverse(x_hat01, sigma_data_v)
                A_clean = A01_v
                psnr_metric.update(A_den, A_clean)
                ssim_metric.update(A_den, A_clean)


        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        print(f"Epoch {epoch+1} — Val PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # ----- Model selection: Pareto + fixed scale TOPSIS -----
        save_flag, state = maybe_update(
            avg_psnr, avg_ssim, PSNR_MIN, PSNR_MAX, SSIM_MIN, SSIM_MAX, epoch + 1,
            w_p=0.5, w_s=0.5, tol=1e-4
        )
        if save_flag:
            torch.save(svnr_model.state_dict(), "MRI_Rician_best.pth")
            epoch_best = epoch
            print(
                f"[{epoch+1}] NEW BEST: PSNR={state['psnr']:.2f} dB, "
                f"SSIM={state['ssim']:.4f}, TOPSIS={state['score']:.4f}"
            )

# [TIMER] 
if device.type == "cuda":
    torch.cuda.synchronize()
total_secs = perf_counter() - total_start # total training time

print("Training completed on:", device)
print(f'Found best model on validation set at epoch {epoch_best+1}')
print("Total training time:", format_dhms(total_secs))

# -------------------------------------------------
#         Average Training Loss per Epoch Plot 
# -------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses,  label='Train MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training Loss per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Optional plot save
plt.savefig("loss_MRI_Rician.png")
print("Saved plot to loss_MRI_Rician.png")

# ============================================================
#                        INFERENCE
# ============================================================

# ----------------------------------------------
# Loading weights and model inizialization
# ----------------------------------------------
svnr_model.load_state_dict(torch.load("MRI_Rician_best.pth",  map_location=device),strict=False) # "MRI_Rician_lowest.pth" for model on lowest loss
svnr_model.eval()                                                                                                                                                  

# ================================================================================
#                     Synthetic Noise Test on 6 noise levels
# ================================================================================
device = next(svnr_model.parameters()).device  # Uses same device as the model

# --- Choose here 2 slices (by dataset index) to visualize ---
VISUAL_IDXS = [85, 520]   

# --- Sigma Values for Test ---
SIGMA_TESTS = [0.028, 0.041, 0.059, 0.086, 0.124, 0.180] 
# note: 0.180 ≈ +12.5% higher than max train sigma value (0.160) -> stress conditions

# -----------------------------------------------------------------------
def to01(x):
    """[-1,1] -> [0,1] via clamp operation """
    return ((x + 1) / 2).clamp(0, 1)

# Load test set on RAM
test_imgs = []
with torch.no_grad():
    for imgs, _ in test_loader:
        test_imgs.append(imgs)
test_imgs = torch.cat(test_imgs, dim=0)  # [N,1,H,W] in [-1,1]
N = test_imgs.size(0)

print(f"Test set with {N} images. Executing {len(SIGMA_TESTS)} tests...\n")

# -----------------------------------------------------------
#                 Loop over each noisy couple
# -----------------------------------------------------------
for test_id, sigma in enumerate(SIGMA_TESTS, 1):
    print(f"=== Test {test_id}: σ={sigma} ===")
    psnr_list, ssim_list = [], []

    # ---- triptych for the two chosen slices ----
    triptychs = []  # (orig_np, noisy_np, den_np), dimensions [H,W]

    with torch.no_grad():
        
        BATCH = getattr(test_loader, 'batch_size', 32)
        for start in range(0, N, BATCH):
            end = min(N, start + BATCH)
            imgs_t = test_imgs[start:end].to(device)  # [B,1,H,W] in [-1,1]
            
            # ----- VST prep -----
            A01_t  = (imgs_t + 1.0) * 0.5
            with torch.no_grad():
                sigma_data_t = estimate_sigma_global_esc(A01_t, q=0.10, ds=4)      # [B,1,1,1]
                z01_t = rice_vst_forward(A01_t, sigma_data_t)                      # [B,1,H,W]

            # 1) Forward SVNR: sample (noisy, xt, tmap)
            noisy, xt, tmap, _ , _= sample_svnr_vst(z01_t, sigma_val, gamma_schedule)
            #print(tmap)

            # 2) Reverse SVNR
            denoised = reverse_svnr(noisy, xt, tmap, svnr_model, gamma_schedule)

            # 3) Per-image metrics (PSNR/SSIM) in [0,1]
            den01 = ((denoised + 1) / 2).clamp(0.0, 1.0)
            A_den_t=rice_vst_inverse(den01, sigma_data_t)
            A_clean_t = A01_t
            gt_np  = A_clean_t.detach().cpu().numpy()   # [B,1,H,W]
            den_np = A_den_t.detach().cpu().numpy()

            for b in range(gt_np.shape[0]):
                g = gt_np[b, 0]; d = den_np[b, 0]
                psnr_list.append(peak_signal_noise_ratio(g, d, data_range=1.0))
                ssim_list.append(structural_similarity(g, d, data_range=1.0))

            # 4) Save triptychs
            for vis_idx in VISUAL_IDXS:
                if start <= vis_idx < end:
                    local = vis_idx - start
                    orig_np  = gt_np[local, 0]
                    noisy_rician = rice_vst_inverse(to01(noisy[local:local+1]), sigma_data_t)
                    noisy_np = noisy_rician.detach().cpu().numpy()[0,0]
                    den_np_s = den_np[local, 0]
                    triptychs.append((orig_np, noisy_np, den_np_s))

    # ======== GLOBAL TEST RESULTS ========
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else float('nan')
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else float('nan')
    print(f"Average PSNR: {mean_psnr:.2f} dB | Average SSIM: {mean_ssim:.4f} on {len(psnr_list)} images")

    # ======== TRIPTYCHS OVER THE CHOSEN 2 SLICES ========
    n_show = min(len(triptychs), len(VISUAL_IDXS))
    for k in range(n_show):
        o, n, d = triptychs[k]
        # PSNR/SSIM COMPUTING
        ps = peak_signal_noise_ratio(o, d, data_range=1.0)
        ss = structural_similarity(o, d, data_range=1.0)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(
            axes,
            [o, n, d],
            ["Original", "Noisy", f"Denoised\nPSNR={ps:.2f}dB SSIM={ss:.4f}"]
        ):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f"Test {test_id}  (σ = {sigma}  |  slice={VISUAL_IDXS[k]})")
        plt.tight_layout()
        plt.savefig(f"triptych_test{test_id}_ESC_slice{VISUAL_IDXS[k]}.png", dpi=300) # Optional figure save
        plt.show()

    
    # ======== TEST PSNR / SSIM HISTOGRAMS ========
    if psnr_list:
        plt.figure(figsize=(5,3))
        plt.hist(psnr_list, bins=30)
        plt.axvline(mean_psnr, color='k', linestyle='--', linewidth=1.5,
            label=f"Mean = {mean_psnr:.2f} dB")
        plt.xlabel('PSNR (dB)'); plt.ylabel('Frequency')
        plt.title(f"PSNR histogram — Test {test_id} (σ = {sigma})")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"hist_psnr_test_MRI_Rician_{test_id}.png", dpi=300) # Optional figure save
        plt.show()

    if ssim_list:
        plt.figure(figsize=(5,3))
        plt.hist(ssim_list, bins=30)
        plt.axvline(mean_ssim, color='k', linestyle='--', linewidth=1.5,
            label=f"Mean = {mean_ssim:.3f}")
        plt.xlabel('SSIM'); plt.ylabel('Frequency')
        plt.title(f"SSIM histogram — Test {test_id} (σ = {sigma})")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"hist_ssim_test_MRI_Rician_{test_id}.png", dpi=300) # Optional figure save
        plt.show()

    print() 

# ============================================================================================
#         Visualization of 20 samples from Test Set - direct denoising (no extra noise)
# ============================================================================================

# Random selection of 20 images from test set
num_samples = 20
dataset = test_loader.dataset
indices = np.random.choice(len(dataset), size=num_samples, replace=False)
samples = torch.stack([dataset[i][0] for i in indices]).to(device)  # [20,1,320,320] in [-1,1]

# Treat the images as observations y (no added noise)
y = samples.clone()                         
y01 = (y + 1.0) * 0.5              # [-1,1] -> [0,1]
# Forward SVNR
# --- per-image (batch) σ_data estimation + VST ---
sigma_data_t = estimate_sigma_global_esc(y01, q=0.10, ds=4)     # [B,1,1,1]
z01        = rice_vst_forward(y01, sigma_data_t)              # [B,1,H,W] in [0,1]
gamma_Tstar = (sigma_data_t ** 2) * z01 
tmap = gamma_to_t_torch(gamma_Tstar, gamma_schedule)                 # [B,1,H,W]

# Reverse SVNR starting from x=y
z11 = (z01 * 2.0 - 1.0)
x = z11.clone()
denoised = reverse_svnr(x, z11, tmap, svnr_model, gamma_schedule)

# Conversion for visualization: (B,H,W,C) in [0,1]
den01 = (denoised + 1.0) * 0.5
A_den_t = rice_vst_inverse(den01, sigma_data_t)
orig_np = y01.detach().cpu().permute(0,2,3,1).numpy()
den_np  = A_den_t.detach().cpu().permute(0,2,3,1).numpy()

# Plot
fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 3))
for i in range(num_samples):
    axes[i,0].imshow(orig_np[i], cmap='gray')
    axes[i,0].set_title('Input (y)')
    axes[i,0].axis('off')
    axes[i,1].imshow(den_np[i], cmap= 'gray')
    axes[i,1].set_title('Denoised')
    axes[i,1].axis('off')
plt.tight_layout()
plt.show()

# Optional figure save
plt.savefig("20_sets_MRI_Rician_esc.png")
print("Saved plot to 20_sets_MRI_Rician_esc.png")

# ============================================================================================
#     Visualization of 4 samples from Test Set - Direct Denoising
# ============================================================================================

# Random selection of 4 images from test set
num_samples = 4
dataset = test_loader.dataset
indices = np.random.choice(len(dataset), size=num_samples, replace=False)
samples = torch.stack([dataset[i][0] for i in indices]).to(device)

# Treat the images as observations y (no added noise)
y = samples.clone()                         
y01 = (y + 1.0) * 0.5              # [-1,1] -> [0,1]
# Forward SVNR
# --- per-image (batch) σ_data estimation + VST ---
sigma_data_t = estimate_sigma_global_esc(y01, q=0.10, ds=4)     # [B,1,1,1]
z01        = rice_vst_forward(y01, sigma_data_t)              # [B,1,H,W] in [0,1]
gamma_Tstar = (sigma_data_t ** 2) * z01 
tmap = gamma_to_t_torch(gamma_Tstar, gamma_schedule)                 # [B,1,H,W]

# Reverse SVNR starting from x=y
z11 = (z01 * 2.0 - 1.0)
x = z11.clone()
denoised = reverse_svnr(x, z11, tmap, svnr_model, gamma_schedule)

# Conversion for visualization: (B,H,W,C) in [0,1]
den01 = (denoised + 1.0) * 0.5
A_den_t = rice_vst_inverse(den01, sigma_data_t)
orig_np = y01.detach().cpu().permute(0,2,3,1).numpy()
den_np  = A_den_t.detach().cpu().permute(0,2,3,1).numpy()

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
    ax_in.imshow(orig_np[i], cmap= 'gray')
    ax_in.set_title(f'Input (y) - slice {indices[i]}', fontsize=10, pad=3)
    ax_in.set_axis_off()

    # Denoised
    ax_dn = fig.add_subplot(gs[r, c_den])
    ax_dn.imshow(den_np[i], cmap= 'gray')
    ax_dn.set_title('Denoised', fontsize=10, pad=3)
    ax_dn.set_axis_off()

# Figure save
fig.subplots_adjust(top=0.95, bottom=0.06, left=0.05, right=0.97)
out_path = "ESC_direct_denoising_4examples.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Saved plot to {out_path}")

plt.show()
plt.close(fig)

# =============================================================================================================
#     Visualization of 4 Direct Denoising samples from Test Set - ESC input + ESC denoising + RSS pseudo-GT
# =============================================================================================================

# Random selection of 4 images from test set (ESC)
num_samples = 4
dataset_esc = test_loader.dataset
dataset_rss = test_loader_rss.dataset
assert len(dataset_esc) == len(dataset_rss), "ESC e RSS dataset devono avere la stessa lunghezza."

indices = np.random.choice(len(dataset_esc), size=num_samples, replace=False)

# Stack ESC and RSS slices (assumo output in [-1,1])
samples_esc = torch.stack([dataset_esc[i][0] for i in indices]).to(device)
samples_rss = torch.stack([dataset_rss[i][0] for i in indices]).to(device)

# === ESC reconstruction denoising ===
y = samples_esc.clone()
y01 = (y + 1.0) * 0.5  # [-1,1] -> [0,1]

sigma_data_t = estimate_sigma_global_esc(y01, q=0.10, ds=4)    # [B,1,1,1]
z01          = rice_vst_forward(y01, sigma_data_t)             # [B,1,H,W] in [0,1]
gamma_Tstar  = (sigma_data_t ** 2) * z01
tmap         = gamma_to_t_torch(gamma_Tstar, gamma_schedule)   # [B,1,H,W]

z11 = (z01 * 2.0 - 1.0)
x = z11.clone()
denoised = reverse_svnr(x, z11, tmap, svnr_model, gamma_schedule)

# Conversion for visualization
den01   = (denoised + 1.0) * 0.5
A_den_t = rice_vst_inverse(den01, sigma_data_t)

orig_np = y01.detach().cpu().permute(0, 2, 3, 1).numpy()
den_np  = A_den_t.detach().cpu().permute(0, 2, 3, 1).numpy()
rss_np  = ((samples_rss + 1.0) * 0.5).detach().cpu().permute(0, 2, 3, 1).numpy()

# --- Figure: 1 tryptich per-row ---
rows, cols = num_samples, 3
fig, axes = plt.subplots(rows, cols, figsize=(cols*5.2, rows*5.0))
plt.subplots_adjust(hspace=0.5)  # vertical space between rows
for i in range(num_samples):
    ax_in, ax_dn, ax_gt = axes[i, 0], axes[i, 1], axes[i, 2]

    # Input (ESC)
    ax_in.imshow(orig_np[i], cmap='gray')
    ax_in.set_title(f'Input (ESC) – slice {indices[i]}', fontsize=16, pad=10)
    ax_in.axis('off')

    # Denoised (ESC)
    ax_dn.imshow(den_np[i], cmap='gray')
    ax_dn.set_title('Denoised (ESC)', fontsize=16, pad=10)
    ax_dn.axis('off')

    # Pseudo-GT (RSS)
    ax_gt.imshow(rss_np[i], cmap='gray')
    ax_gt.set_title('Pseudo-GT (RSS)', fontsize=16, pad=10)
    ax_gt.axis('off')

plt.tight_layout()
out_path = "ESC_vs_Denoised_vs_RSS.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05) # Optional: figure save
print(f"Saved plot to {out_path}")
plt.show()
plt.close(fig)

