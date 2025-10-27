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

# --- PyTorch ecosystem ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchinfo import summary
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# --- Other framework ---
from tensorflow.keras.datasets import mnist as keras_mnist

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
batch_size      = 128
lr              = 1e-4
num_epochs      = 200

# Model
beta_start = 1e-4
beta_end = 0.02
T               = 1000
lambda_val      = 5
sigma_r_train   = (1e-3, 5e-1)
sigma_s_train   = (1e-4, 3e-1)
sigma_r_val     = 0.1005
sigma_s_val     = 0.02505

# Deterministic Data Loading
SEED = 12345
num_workers = 4
val_frac = 0.1    # validation fraction

# PSNR and SSIM Normalization Bounds
PSNR_MIN, PSNR_MAX = 10.0, 40.0     # dB   (clip for normalization)
SSIM_MIN, SSIM_MAX = 0.50, 1.00     # unit (clip for normalization)

# Pareto tolerance epsilons
EPS_SSIM = 7e-4
EPS_PSNR = 0.3

# ============================================================
# MNIST (Keras) - Deterministic Data loading
# ============================================================

# ------------------------------
# Seed
# ------------------------------ 
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def seed_worker(worker_id: int):
    # Deterministic RNG for each worker
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------------------------
# Separated Generators  (split/loader)
# ------------------------------------
g_split = torch.Generator().manual_seed(SEED)      # random_split
g_train = torch.Generator().manual_seed(SEED)      # DataLoader train (shuffle)
g_val   = torch.Generator().manual_seed(SEED + 1)  # DataLoader val
g_test  = torch.Generator().manual_seed(SEED + 2)  # DataLoader test

# ------------------------------
# Loader parameters
# ------------------------------
pin_memory  = False
if torch.cuda.is_available():
    pin_memory = True

# ------------------------------
# Dataset 
# ------------------------------
class KerasMNISTDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        X : np.ndarray (N, 28, 28)  0-255 uint8 or 0-1 float
        y : np.ndarray (N,)
        transform : torchvision.transforms  (ToTensor -> Normalize)
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        # assures uint8 0-255 per PIL
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')  # grayscale

        if self.transform:
            img = self.transform(img)         # tensor 1x28x28 in [-1,1]

        label = int(self.y[idx])
        return img, label

# ------------------------------
# Transforms
# ------------------------------
base_transforms = transforms.Compose([
    transforms.ToTensor(),               # -> [0,1]
    transforms.Normalize((0.5,), (0.5,)) # -> [-1,1]
])

# ------------------------------
# Loading MNIST Data from Keras
# ------------------------------
(X_train, y_train), (X_test, y_test) = keras_mnist.load_data()

train_full_ds = KerasMNISTDataset(X_train, y_train, transform=base_transforms)
test_ds       = KerasMNISTDataset(X_test,  y_test,  transform=base_transforms)

# ------------------------------
# Deterministic Split train/val
# ------------------------------
n_val   = int(len(train_full_ds) * val_frac)
n_train = len(train_full_ds) - n_val
train_ds, val_ds = random_split(train_full_ds, [n_train, n_val], generator=g_split)

# ------------------------------------------
# DataLoader (with generator + seed_worker)
# ------------------------------------------
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,              # deterministic shuffle by generator
    drop_last=True,         
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g_train,
    persistent_workers=(num_workers > 0),
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,         
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g_val,
    persistent_workers=(num_workers > 0),
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,            
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g_test,
    persistent_workers=(num_workers > 0),
)

print(f" Ready (MNIST): train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# ============================================================
#                        Network
# ============================================================
class LightUNetSVNR(nn.Module):
    def __init__(self, base_ch=64, time_dim=128, dropout_p=0.2):
        super().__init__()

        # 1) Time-embedding MLP (global)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
        )

        # 2) Encoder
        # inc: input [y, x] -> 2 channels 
        self.inc   = self._conv_block(2, base_ch, dropout_p)
        # down1: [x1, temb, tmap_32] => base_ch + time_dim + 1
        self.down1 = self._conv_block(base_ch + time_dim + 1, base_ch * 2, dropout_p, down=True)
        # down2: [x2, temb_2, tmap_16] => base_ch*2 + time_dim + 1
        self.down2 = self._conv_block(base_ch * 2 + time_dim + 1, base_ch * 4, dropout_p, down=True)

        # 3) Bottleneck:  [x3, temb_4, tmap_8] => base_ch*4 + time_dim + 1
        self.bot   = self._conv_block(base_ch * 4 + time_dim + 1, base_ch * 4, dropout_p)

        # 4) Decoder
        self.up1_tr = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        # up1: [u1_tr, x2, temb_2, tmap_16] => base_ch*2 + base_ch*2 + time_dim + 1
        self.up1    = self._conv_block(base_ch*2 + base_ch*2 + time_dim + 1, base_ch*2, dropout_p)

        self.up2_tr = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        # up2: [u2_tr, x1, temb, tmap_32] => base_ch + base_ch + time_dim + 1
        self.up2    = self._conv_block(base_ch + base_ch + time_dim + 1, base_ch, dropout_p)

        # 5) Output
        self.out    = nn.Conv2d(base_ch, 1, kernel_size=1)

    def _conv_block(self, in_ch, out_ch, p, down=False):
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        layers += [
            nn.BatchNorm2d(out_ch, momentum=0.9, eps=1e-5),
            nn.GELU(),
            nn.Dropout2d(p),
        ]
        return nn.Sequential(*layers)

    def forward(self, y, x, tmap):
        B, C, H, W = x.shape

        # --- global time embedding ---
        t = tmap.view(B, 1, -1).mean(-1, keepdim=True)        # [B,1]
        temb = self.time_mlp(t)                                # [B, time_dim]
        temb = temb.view(B, -1, 1, 1).expand(-1, -1, H, W)     # [B, time_dim, H, W]

        # --- scaled-spatial time maps: tmap_32 (H), tmap_16 (H/2), tmap_8 (H/4) ---
        tmap_32 = tmap                                         # [B,1,H,W]
        tmap_16 = F.avg_pool2d(tmap_32, kernel_size=2)         # [B,1,H/2,W/2]
        tmap_8  = F.avg_pool2d(tmap_16, kernel_size=2)         # [B,1,H/4,W/4]

        # --- encoder path ---
        x0 = torch.cat([y, x], dim=1)                          # [B,2,H,W]
        x1 = self.inc(x0)                                      # [B, base_ch, H, W]

        d1_in = torch.cat([x1, temb, tmap_32], dim=1)          # +1 channel
        x2 = self.down1(d1_in)                                 # [B, base_ch*2, H/2, W/2]

        temb2 = temb[:, :, ::2, ::2]                           # [B, time_dim, H/2, W/2]
        d2_in = torch.cat([x2, temb2, tmap_16], dim=1)         # +1 channel
        x3 = self.down2(d2_in)                                 # [B, base_ch*4, H/4, W/4]

        temb4 = temb2[:, :, ::2, ::2]                          # [B, time_dim, H/4, W/4]
        bot_in = torch.cat([x3, temb4, tmap_8], dim=1)         # +1 channel
        b  = self.bot(bot_in)                                  # [B, base_ch*4, H/4, W/4]

        # --- decoder path ---
        u1_tr = self.up1_tr(b)                                 # [B, base_ch*2, H/2, W/2]
        u1_in = torch.cat([u1_tr, x2, temb2, tmap_16], dim=1)  # +1 channel
        u1 = self.up1(u1_in)                                   # [B, base_ch*2, H/2, W/2]

        u2_tr = self.up2_tr(u1)                                # [B, base_ch, H, W]
        u2_in = torch.cat([u2_tr, x1, temb, tmap_32], dim=1)   # +1 channel
        u2 = self.up2(u2_in)                                   # [B, base_ch, H, W]

        return self.out(u2)                                     # [B,1,H,W]

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
    topsis_score   = d_worst / (d_worst + d_best + eps)  # + eps avoids division by 0 # in [0,1], higher = better
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
            return val.view(-1, *[1]*(imgs.ndim-1)) if val.numel() > 1 else \
                   val.view(1,1,1,1).expand(B,1,1,1)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            lo, hi = map(float, val)
            return torch.empty(B,1,1,1, device=dev, dtype=dtype).uniform_(lo, hi)
        return torch.tensor(float(val), device=dev, dtype=dtype).view(1,1,1,1).expand(B,1,1,1)

    sigma_r = prep(sigma_r)
    sigma_s = prep(sigma_s)

    # 1) sigma_p (shot+read) and gamma_T
    imgs01 = (imgs + 1.0) * 0.5                            # [-1,1] -> [0,1]
    if test:
        imgs01 = imgs01.clamp(0.0, 1.0)                    # only for test

    sigma_p = torch.sqrt(sigma_r**2 + sigma_s**2 * imgs01).clamp(min=1e-3)
    gamma_T = sigma_p**2                                   # [B,1,H,W]

    # 2) building noisy image -> clamp min to 0
    eps_T = torch.randn_like(imgs)
    noisy01 = (imgs01 + sigma_p * eps_T).clamp(min=0.0)   #.clamp(0.,1.)
    noisy = noisy01 * 2.0 - 1.0                            # back to [-1,1]

    # 3) t^_map: gamma_T -> t_map, then casual shift t0 and residual t^ (tmap)
    tmap_full = gamma_to_t_torch(gamma_T, gamma_table)     # t* per-pixel
    t0 = torch.rand((), device=dev, dtype=dtype) * tmap_full.max()   # scalar U(0, max t*)
    tmap = (tmap_full - t0).clamp_min(0.0)                 # residual t^

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
    xt01 = (imgs01 + torch.sqrt(gamma_t) * eps_tilde).clamp(min=0.0)   # .clamp(0.,1.)
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
svnr_model =  LightUNetSVNR().to(device)

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
    weight_decay=1e-4    # moderate L2 to stop overfitting
) # uses decoupled weight decay
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
scaler = GradScaler()

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
    desc = f"Epoch {epoch+1}/{num_epochs}"
    pbar = tqdm(train_loader, desc=desc, leave=True)
    running_loss = 0.0

    # ----- Training -----
    svnr_model.train()
    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True) 
        
        # ----- Forward SVNR -----
        noisy, xt, tmap, eps_tilde, sigma_p = sample_svnr(
            imgs, sigma_r_train, sigma_s_train, gamma_schedule
        )

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
    print(f"Epoch {epoch+1} - Train loss: {epoch_train_loss:.4f}")

    # Saves model with minimum training loss
    if epoch_train_loss < best_train_loss:
        best_train_loss = epoch_train_loss
        torch.save(svnr_model.state_dict(), "SVNR_MNIST_unet_lowest.pth")

    scheduler.step()

    # ----- Validation every 10 epochs -----
    if (epoch + 1) % 10 == 0:
        svnr_model.eval()
        psnr_metric.reset(); ssim_metric.reset()

        with torch.no_grad():
            for imgs_v, _ in val_loader:
                imgs_v = imgs_v.to(device, non_blocking=True)  
                
                # ----- Forward SVNR -----
                noisy_v, x_v, tmap_v, _, _ = sample_svnr(
                    imgs_v, sigma_r_val, sigma_s_val, gamma_schedule
                )

                # ----- Reverse SVNR -----
                x = reverse_svnr(noisy_v, x_v, tmap_v, svnr_model, gamma_schedule)

                den_v = ((x + 1) / 2).clamp(0.0, 1.0)      
                gt_v  = ((imgs_v + 1) / 2).clamp(0.0, 1.0) 

                psnr_metric.update(den_v, gt_v)
                ssim_metric.update(den_v, gt_v)

        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        print(f"Epoch {epoch+1} - Val PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # ----- Model selection: Pareto + fixed scale TOPSIS -----
        save_flag, state = maybe_update(avg_psnr, avg_ssim, PSNR_MIN, PSNR_MAX, SSIM_MIN, SSIM_MAX, epoch + 1,
                                        w_p=0.5, w_s=0.5, tol=1e-4)
        if save_flag:
            torch.save(svnr_model.state_dict(), "SVNR_MNIST_unet_best.pth")
            print(f"[{epoch+1}] NEW BEST: PSNR={state['psnr']:.2f} dB, "
                  f"SSIM={state['ssim']:.4f}, TOPSIS={state['score']:.4f}")

print("Training completed on:", device)

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

# Optional: plot save
plt.savefig("loss.png")
print("Saved plot to loss.png")

# ============================================================
#                        INFERENCE
# ============================================================

# ----------------------------------------------
# Loading weights and model inizialization
# ----------------------------------------------
svnr_model.load_state_dict(torch.load("SVNR_MNIST_unet_best.pth", map_location=device)) # SVNR_MNIST_unet_lowest.pth for best on training loss
svnr_model.eval()                                                                       # SVNR_MNIST_unet_best.pth for best on validation                                                                               

# ====================================================================
#       Multiple Random Images Evaluation - Synthetic Noise Test
#     Using the fixed sigma_r_val and sigma_s_val to corrupt images
# ====================================================================
num_samples = 3
indices = random.sample(range(len(test_ds)), num_samples)

# Figure setup
fig = plt.figure(figsize=(9, 2.6 * num_samples), constrained_layout=False)
gs = fig.add_gridspec(num_samples, 3, wspace=0.05, hspace=0.42)

for i, idx in enumerate(indices):
    # ---- Load image ----
    img, _ = test_ds[idx]
    img = img.unsqueeze(0).to(device)

    # ---- Forward/Reverse SVNR ----
    noisy, xt, tmap, _, _ = sample_svnr(img, sigma_r_val, sigma_s_val, gamma_schedule)
    denoised = reverse_svnr(noisy, xt, tmap, svnr_model, gamma_schedule)

    # ---- Tensors in numpy [0,1] ----
    noisy_np = (noisy.detach().cpu().squeeze().numpy() + 1) / 2
    orig_np = (img.detach().cpu().squeeze().numpy() + 1) / 2
    den_np = (denoised.detach().cpu().squeeze().numpy() + 1) / 2
    noisy_np = noisy_np.clip(0, 1)
    orig_np = orig_np.clip(0, 1)
    den_np = den_np.clip(0, 1)

    # ---- Metrics ----
    psnr = peak_signal_noise_ratio(orig_np, den_np, data_range=1.0)
    ssim = structural_similarity(orig_np, den_np, data_range=1.0)
    t_min, t_max = round(tmap.min().item(), 2), round(tmap.max().item(), 2)

    # ---- Axes for the row ----
    ax0 = fig.add_subplot(gs[i, 0])
    ax1 = fig.add_subplot(gs[i, 1])
    ax2 = fig.add_subplot(gs[i, 2])

    # ---- Per-row header (like suptitle) ABOVE the middle axis ----
    ax1.text(
        0.7, 1.16,
        f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f} | Time map range: [{t_min}, {t_max}]",
        transform=ax1.transAxes, ha='center', va='bottom', fontsize=12
    )

    # ---- Column labels ----
    ax0.set_title('Original', fontsize=10.5, pad=4)
    ax1.set_title('Noisy', fontsize=10.5, pad=4)
    ax2.set_title('Denoised', fontsize=10.5, pad=4)

    # ---- Images ----
    ax0.imshow(orig_np, cmap='gray')
    ax1.imshow(noisy_np, cmap='gray')
    ax2.imshow(den_np, cmap='gray')
    for a in (ax0, ax1, ax2):
        a.set_axis_off()

fig.subplots_adjust(top=0.98, bottom=0.03, left=0.02, right=0.98, hspace=0.45, wspace=0.05)
fig.savefig("mnist_3samples_compact.png", dpi=300, bbox_inches="tight", pad_inches=0.03)  # Optional: figure save
plt.show()
plt.close(fig)

# =========================================================================
#            Model evaluation on entire test set (no visualization)
# =========================================================================
psnr_list = []
ssim_list = []

with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)

        # 1) Forward SVNR: sample
        noisy, xt, tmap, _, _ = sample_svnr(imgs, sigma_r_val, sigma_s_val, gamma_schedule)
        
        # 2) Reverse SVNR
        denoised= reverse_svnr(noisy, xt, tmap, svnr_model, gamma_schedule)

        # 3) Batch-wise metrics computation
        orig_np = (imgs.cpu().numpy() + 1) / 2  # [B,1,H,W] in [0,1]
        den_np  = (denoised.cpu().numpy() + 1) / 2

        # optional
        orig_np = orig_np.clip(0, 1)
        den_np  = den_np.clip(0, 1)
        
        for o, d in zip(orig_np, den_np):
            psnr_list.append(peak_signal_noise_ratio(o[0], d[0], data_range=1.0))
            ssim_list.append(structural_similarity(o[0], d[0], data_range=1.0))

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
fig.savefig("psnr_ssim_histograms_mnist.png", dpi=300, bbox_inches="tight")
print("Saved plot to psnr_ssim_histograms_mnist.png")


# ============================================================================================
#         Visualization of 20 samples from Test Set - Direct Denoising (no extra noise)
# ============================================================================================

# Random selection of 20 images from test set
num_samples = 20
dataset = test_loader.dataset
indices = np.random.choice(len(dataset), size=num_samples, replace=False)
samples = torch.stack([dataset[i][0] for i in indices]).to(device)  # [20,3,28,28] in [-1,1]

# Treat the images as observations y (no added noise)
y = samples.clone()                         # [-1,1]

# Forward SVNR
xt, tmap = sample_svnr(y, sigma_r_val, sigma_s_val, gamma_schedule, test=True)

# Reverse SVNR starting from x=y
x        = y.clone()
denoised = reverse_svnr(x, xt, tmap, svnr_model, gamma_schedule)

# Conversion for visualization: (B,H,W,C) in [0,1]
orig_np = ((samples.detach().cpu().permute(0,2,3,1).numpy()) + 1) / 2
den_np  = ((denoised.detach().cpu().permute(0,2,3,1).numpy()) + 1) / 2

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

# Optional: plot save
plt.savefig("MNIST_direct_denoising.png")
print("Saved plot to MNIST_direct_denoising.png")

# ============================================================================================
#     Visualization of 4 samples from Test Set (4x4 grid) - Direct Denoising
# ============================================================================================

# Random selection of 4 images from test set
num_samples = 4
dataset = test_loader.dataset
indices = np.random.choice(len(dataset), size=num_samples, replace=False)
samples = torch.stack([dataset[i][0] for i in indices]).to(device)

# Direct denoising (no extra noise)
y = samples.clone()
xt, tmap = sample_svnr(y, sigma_r_val, sigma_s_val, gamma_schedule, test=True)
denoised = reverse_svnr(y.clone(), xt, tmap, svnr_model, gamma_schedule)

# to numpy [0,1]
orig_np = ((samples.detach().cpu().permute(0,2,3,1).numpy()) + 1) / 2
den_np  = ((denoised.detach().cpu().permute(0,2,3,1).numpy()) + 1) / 2

# --- Figure ---
fig = plt.figure(figsize=(9.2, 5.0))
# Grid: [Input1, Denoised1, SPACER, Input2, Denoised2]
gs = gridspec.GridSpec(
    2, 5,
    width_ratios=[1, 1, 0.15, 1, 1],
    wspace=0.08,
    hspace=0.28
)

for i in range(num_samples):
    r = i // 2
    c_in  = 0 if (i % 2)==0 else 3
    c_den = c_in + 1

    # Input (y)
    ax_in = fig.add_subplot(gs[r, c_in])
    ax_in.imshow(orig_np[i], cmap='gray')
    ax_in.set_title('Input (y)', fontsize=10, pad=3)
    ax_in.set_axis_off()

    # Denoised
    ax_dn = fig.add_subplot(gs[r, c_den])
    ax_dn.imshow(den_np[i], cmap='gray')
    ax_dn.set_title('Denoised', fontsize=10, pad=3)
    ax_dn.set_axis_off()

# Optional: save figure
fig.subplots_adjust(top=0.95, bottom=0.06, left=0.05, right=0.97)
fig.savefig("MNIST_direct_denoising_4examples.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
print("Saved plot to MNIST_direct_denoising_4examples.png")
plt.show()
plt.close(fig)


