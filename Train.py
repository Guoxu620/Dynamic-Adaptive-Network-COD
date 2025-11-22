# train_with_aug.py  —— 在训练集加入数据增强（flip/rotate/color jitter），验证集不增强
import os
import argparse
import logging
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import random
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.utils import clip_gradient
from model.DABNet import DABNet
from metrics import Smeasure


# =========================
# 数据集：训练集(带增强)/验证集(无增强)
# =========================
class AugDataset(Dataset):
    """
    pairs: [(img_path, gt_path), ...]
    image_size: 统一resize到正方形（如704）
    augment=True 时，对 image/gt 做成对的空间增强（翻转/旋转）；颜色增强只作用于 image
    """
    def __init__(self, pairs, image_size=704, augment=False):
        self.pairs = pairs
        self.size = image_size
        self.augment = augment

        # 最终尺寸对齐 + 标准化（ImageNet）
        self.img_transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
            transforms.ToTensor(),  # -> [0,1]
        ])

        # 颜色抖动（只对图像）
        self.color_jit = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        gt  = Image.open(gt_path).convert("L")

        if self.augment:
            # ---- 随机水平翻转（成对）----
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                gt  = gt.transpose(Image.FLIP_LEFT_RIGHT)

            # ---- 小角度随机旋转（成对）----
            if random.random() < 0.30:
                angle = random.randint(-15, 15)
                img = img.rotate(angle, resample=Image.BICUBIC)
                gt  = gt.rotate(angle, resample=Image.NEAREST)

            # ---- 颜色增强（仅图像）----
            if random.random() < 0.80:
                img = self.color_jit(img)

        img = self.img_transform(img)
        gt  = self.gt_transform(gt)
        return img, gt


def split_dataset(image_root, gt_root, image_size=704, val_ratio=0.1):
    imgs = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    gts  = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))])

    # 仅保留尺寸匹配的一一对应样本
    pairs = []
    for ip, gp in zip(imgs, gts):
        try:
            if Image.open(ip).size == Image.open(gp).size:
                pairs.append((ip, gp))
        except Exception:
            continue

    train_pairs, val_pairs = train_test_split(pairs, test_size=val_ratio, random_state=10042)

    train_set = AugDataset(train_pairs, image_size=image_size, augment=True)   # ✅ 训练集增强
    val_set   = AugDataset(val_pairs,   image_size=image_size, augment=False)  # 验证集不增强
    return train_set, val_set


# =========================
# 你原来的 Utils/损失/EMA/验证，保持不变
# =========================
def seed_everything(seed: int = 10042):
    import numpy as _np
    import torch as _torch
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.benchmark = True  # speed

@torch.no_grad()
def _safe_forward_eval(model, imgs):
    """
    兼容两种签名：
      - model(imgs, return_aux=False)  -> returns (None_or_stages, preds)
      - model(imgs)                    -> returns (stages, preds)
    """
    try:
        out = model(imgs, return_aux=False)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            _, preds = out
            return preds
        return out
    except TypeError:
        out = model(imgs)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            _, preds = out
            return preds
        return out

def structure_loss(pred, mask):
    # Weighted BCE + Weighted IoU（你的实现）
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)

    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1.0) / (union - inter + 1.0)

    return (wbce + wiou).mean()

class ModelEMA:
    """简单的EMA（按iteration更新）"""
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy_model(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))

def deepcopy_model(model: nn.Module) -> nn.Module:
    import copy
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_one_epoch(train_loader, model, optimizer, epoch, opt, total_step, writer, scaler, ema):
    model.train()
    device = next(model.parameters()).device
    total_loss, total_stage, total_map = 0.0, 0.0, 0.0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, gts) in enumerate(train_loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=opt.amp):
            # forward（训练期需要深监督分支）
            try:
                stage_pre, pre = model(imgs)
            except TypeError:
                _, pre = model(imgs, return_aux=False)
                stage_pre = []

            if isinstance(stage_pre, torch.Tensor):
                stage_pre = [stage_pre]

            stage_loss_list = [structure_loss(out, gts) for out in stage_pre]
            # 与原逻辑保持一致：第 i 个深监督乘 (0.2 * i)
            stage_loss = sum((0.2 * i) * l for i, l in enumerate(stage_loss_list))
            map_loss = structure_loss(pre, gts)
            loss = stage_loss + map_loss

        scaler.scale(loss).backward()

        if step % opt.grad_accum == 0:
            if opt.clip is not None and opt.clip > 0:
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, opt.clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += loss.item()
        total_stage += stage_loss.item() if torch.is_tensor(stage_loss) else float(stage_loss)
        total_map += map_loss.item()

        if step % opt.log_interval == 0 or step == total_step:
            lr = optimizer.param_groups[0]['lr']
            eta = (time.time() - t0) / step * (total_step - step + 1)
            msg = (f"[{datetime.now()}] => [Epoch {epoch:03d}/{opt.epoch:03d}] "
                   f"Step {step:04d}/{total_step:04d} | "
                   f"Loss: {loss.item():.4f} (stage {stage_loss:.4f}, map {map_loss:.4f}) | LR {lr:.2e} | ETA {eta/60:.1f}m")
            print(msg)
            logging.info("#TRAIN#: " + msg)

    mean_loss = total_loss / total_step
    writer.add_scalar("Train/Loss", mean_loss, global_step=epoch)
    writer.add_scalar("Train/StageLoss", total_stage / max(1, len(train_loader)), global_step=epoch)
    writer.add_scalar("Train/MapLoss", total_map / max(1, len(train_loader)), global_step=epoch)
    return mean_loss


@torch.no_grad()
def validate(val_loader, model, use_ema=False, ema=None):
    device = next(model.parameters()).device
    mdl = ema.ema if (use_ema and ema is not None) else model
    mdl.eval()
    s_metric = Smeasure()

    for imgs, gts in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        preds = _safe_forward_eval(mdl, imgs)
        preds = torch.sigmoid(preds)

        for pred, gt in zip(preds, gts):
            pred_np = (pred.squeeze().detach().cpu().clamp_(0, 1).numpy() * 255).astype(np.uint8)
            gt_np = (gt.squeeze().detach().cpu().clamp_(0, 1).numpy() * 255).astype(np.uint8)
            s_metric.step(pred_np, gt_np)

    sm_result = s_metric.get_results()['sm']
    return sm_result


@torch.no_grad()
def validate_loss(val_loader, model, use_ema=False, ema=None):
    device = next(model.parameters()).device
    mdl = ema.ema if (use_ema and ema is not None) else model
    mdl.eval()
    total_loss = 0.0
    n = 0
    for imgs, gts in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        preds = _safe_forward_eval(mdl, imgs)
        loss = structure_loss(preds, gts)
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--trainsize', type=int, default=704)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--train_path', type=str, default='your dataset path')
    parser.add_argument('--save_path', type=str, default='your save path')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--seed', type=int, default=10042)
    parser.add_argument('--amp', action='store_true', help='use mixed precision (AMP)')
    parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='linear warmup epochs (0=disable)')
    parser.add_argument('--ema', action='store_true', help='enable EMA of weights for eval')
    parser.add_argument('--early_stop', type=int, default=0, help='patience on S-measure (0=disable)')
    parser.add_argument('--log_interval', type=int, default=20)
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(opt.save_path, 'log.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("DABNet RGB-Only Training (with data augmentation on train set)")

    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DABNet().to(device)

    # Optimizer
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

    # Warmup + Cosine
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    if opt.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=opt.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, opt.epoch - opt.warmup_epochs), eta_min=1e-7)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[opt.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, opt.epoch), eta_min=1e-7)

    # Data
    train_image_root = os.path.join(opt.train_path, 'Imgs')
    train_gt_root    = os.path.join(opt.train_path, 'GT')
    train_set, val_set = split_dataset(train_image_root, train_gt_root, image_size=opt.trainsize, val_ratio=0.1)

    train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=opt.batchsize, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    total_step = len(train_loader)
    writer = SummaryWriter(os.path.join(opt.save_path, "SummaryWriter"))

    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    ema = ModelEMA(model) if opt.ema else None

    best_s = -1.0
    best_epoch = 0
    no_improve = 0

    print('-------------------- Start Training ----------------------')
    for epoch in range(1, opt.epoch + 1):
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, opt, total_step, writer, scaler, ema)

        # validate（若开启 EMA，则用 EMA 模型评估）
        val_s = validate(val_loader, model, use_ema=opt.ema, ema=ema)
        val_loss = validate_loss(val_loader, model, use_ema=opt.ema, ema=ema)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Val/Smeasure", val_s, global_step=epoch)
        writer.add_scalar("Val/Loss", val_loss, global_step=epoch)
        writer.add_scalar("Train/LR", current_lr, global_step=epoch)

        print(f'[Validation] Epoch {epoch:03d} | S-measure: {val_s:.4f} | Val Loss: {val_loss:.4f} | LR {current_lr:.2e}')
        logging.info(f'#VAL#: Epoch {epoch:03d}, S-measure: {val_s:.4f}, Loss: {val_loss:.4f}, LR {current_lr:.2e}')

        # 保存最优
        if val_s > best_s:
            best_s = val_s
            best_epoch = epoch
            torch.save((ema.ema if (opt.ema and ema is not None) else model).state_dict(),
                       os.path.join(opt.save_path, 'best_model.pth'))
            print(f'✅ Best model saved at epoch {epoch}, S-measure improved to {val_s:.4f}')
            no_improve = 0
        else:
            no_improve += 1

        # 每N个epoch保存一次
        if epoch % opt.epoch_save == 0:
            ckpt_name = f'model_epoch_{epoch}.pth'
            torch.save(model.state_dict(), os.path.join(opt.save_path, ckpt_name))

        # 早停（可选）
        if opt.early_stop > 0 and no_improve >= opt.early_stop:
            print(f'⏹️ Early stopping at epoch {epoch} (patience={opt.early_stop}). Best S: {best_s:.4f} at epoch {best_epoch}.')
            break

    print(f'🎯 Training Finished! Best epoch: {best_epoch}, Best S-measure: {best_s:.4f}')
    writer.close()


if __name__ == "__main__":
    main()
