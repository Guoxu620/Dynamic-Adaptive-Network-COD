import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.RISNet import RISNet
from utils.dataloader import test_dataset  # 已是 RGB-only 版本

# ----------------------------
# 解析命令行参数
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size default 704')
parser.add_argument('--pth_path', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--test_path', type=str, required=True, help='path to test dataset root, should contain COD-10K/Test/')
opt = parser.parse_args()

# ----------------------------
# 推理主流程
# ----------------------------
for _data_name in ['CAMO' ]:
    data_path = os.path.join(opt.test_path, _data_name, 'Test')
    save_path = os.path.join('./results', _data_name)
    os.makedirs(save_path, exist_ok=True)

    # 初始化模型
    model = RISNet()
    model.load_state_dict(torch.load(opt.pth_path), strict=False)
    model.cuda()
    model.eval()

    # 设置路径
    image_root = os.path.join(data_path, 'Imgs')
    gt_root = os.path.join(data_path, 'GT')
    print('root', image_root, gt_root)

    # 加载测试集（RGB-only）
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    print('**** Total samples:', test_loader.size)

    # 使用 tqdm 包装循环，加进度条
    for _ in tqdm(range(test_loader.size), desc='Processing'):
        image, gt, name = test_loader.load_data()
        tqdm.write(f"*** name: {name}")

        # 预处理
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # 推理（只用 final_pred / P2）
        with torch.no_grad():
            _, P2 = model(image)
            res = F.interpolate(P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

        # 合法性检查
        if res.ndim != 2 or res.shape[0] == 0 or res.shape[1] == 0:
            tqdm.write(f"[跳过] {name} 输出尺寸异常: {res.shape}")
            continue
        if np.isnan(res).any() or np.isinf(res).any():
            tqdm.write(f"[跳过] {name} 输出包含 NaN 或 Inf")
            continue

        # 与验证保持一致：不做 per-image min–max，直接保存 sigmoid*255
        res = (res * 255).astype(np.uint8)
        save_file = os.path.join(save_path, name)
        try:
            cv2.imwrite(save_file, res)
        except Exception as e:
            tqdm.write(f"[写入失败] {name} - {e}")