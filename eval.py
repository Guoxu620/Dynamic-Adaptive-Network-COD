import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import metrics


def Borders_Capture(gt, pred, dksize=15):
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros_like(gt)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    kernel = np.ones((dksize, dksize), np.uint8)
    img_dilate = cv2.dilate(img, kernel)

    res = cv2.bitwise_and(img_dilate, gt)
    b, g, r = cv2.split(res)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    merge = cv2.merge((b, g, r, alpha))

    resp = cv2.bitwise_and(img_dilate, pred)
    b, g, r = cv2.split(resp)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    mergep = cv2.merge((b, g, r, alpha))

    merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
    mergep = cv2.cvtColor(mergep, cv2.COLOR_RGB2GRAY)
    return merge, mergep, np.sum(img_dilate) / 255


def eval(args, dataset):
    FM = metrics.Fmeasure_and_FNR()
    WFM = metrics.WeightedFmeasure()
    SM = metrics.Smeasure()
    EM = metrics.Emeasure()
    MAE = metrics.MAE()
    BR_MAE = metrics.MAE()
    BR_wF = metrics.WeightedFmeasure()

    gt_root = os.path.join(args.GT_root, dataset, 'Test', 'GT')

    pred_root = os.path.join(args.pred_root, dataset)

    gt_name_list = sorted([f for f in os.listdir(pred_root) if f.endswith(('.png', '.jpg'))])
    if len(gt_name_list) == 0:
        print(f"[❌错误] 没有在 {pred_root} 中找到预测结果文件")
        return

    print(f"\n👉 评估数据集：{dataset}")
    for gt_name in tqdm(gt_name_list, desc=f"Evaluating {dataset}", ncols=70):
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)

        if not os.path.exists(gt_path):
            print(f"[⚠️警告] 找不到对应 GT 文件：{gt_path}，已跳过")
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            print(f"[⚠️警告] 无法读取图像：{gt_name}，已跳过")
            continue

        if gt.shape != pred.shape:
            pred = cv2.resize(pred, gt.shape[::-1])
            cv2.imwrite(pred_path, pred)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

        if args.BR == 'on':
            BR_gt, BR_pred, area = Borders_Capture(cv2.imread(gt_path), cv2.imread(pred_path), int(args.br_rate))
            BR_MAE.step(pred=BR_pred, gt=BR_gt, area=area)
            BR_wF.step(pred=BR_pred, gt=BR_gt)

    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]

    record = [
        f"Model:{args.model}, Dataset:{dataset} ||",
        f"Smeasure:{sm:.3f}; meanEm:{'-' if em['curve'] is None else em['curve'].mean():.3f};",
        f"wFmeasure:{wfm:.3f}; MAE:{mae:.3f}; fnr:{fnr:.3f};",
        f"adpEm:{em['adp']:.3f}; maxEm:{'-' if em['curve'] is None else em['curve'].max():.3f};",
        f"adpFm:{fm['adp']:.3f}; meanFm:{fm['curve'].mean():.3f}; maxFm:{fm['curve'].max():.3f};"
    ]

    if args.BR == 'on':
        BRmae = BR_MAE.get_results()['mae']
        BRwF = BR_wF.get_results()['wfm']
        record.append(f"BR{args.br_rate}_mae:{BRmae:.3f}; BR{args.br_rate}_wF:{BRwF:.3f}")

    eval_record = ' '.join(record)
    print(eval_record)
    print("#" * 50)

    txt = args.record_path or 'output/eval_record.txt'
    os.makedirs(os.path.dirname(txt), exist_ok=True)
    with open(txt, 'a') as f:
        f.write(eval_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='DABNet')
    parser.add_argument("--pred_root", default='./results')
    parser.add_argument("--GT_root", default='../')
    parser.add_argument("--record_path", default='output/eval_record.txt')
    parser.add_argument("--BR", default='on')
    parser.add_argument("--br_rate", default=15)
    parser.add_argument("--datasets", nargs='+', default=['COD10K', 'CAMO', 'CHAMELEON'],
                        help="多个数据集名，例如：--datasets COD10K CAMO")

    args = parser.parse_args()

    for dataset in args.datasets:
        eval(args, dataset)
