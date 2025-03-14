import torch
import numpy as np
import cv2 as cv
import PIL.Image as Image


def show_img(img,img_name='img', waitKey=1):
    if img.shape[-1] == 1:
        img = (img - img.min()) / (img.max() - img.min())
        img = img.repeat(1, 1, 3)
        # img = torch.where(img > 0.25, img, 0.0)
    img = img.float().detach().cpu() * 255
    img = np.uint8(img)
    cv.imshow(img_name, img[:, :, ::-1])
    cv.waitKey(waitKey)
    return img


def show_heatmap(img, map, img_name='img', waitKey=1):
    map = (map - map.min()) / (map.max() - map.min())
    map = map.repeat(1, 1, 3).float().detach().cpu() * 255
    map = np.uint8(map)

    img = img.float().detach().cpu() * 255
    img = np.uint8(img)

    map = cv.applyColorMap(map, cv.COLORMAP_HOT)
    # map = cv.cvtColor(map, cv.COLOR_RGB2BGR)

    img = cv.addWeighted(img, 0.1, map, 0.9, 0)

    cv.imshow(img_name, img)
    cv.waitKey(waitKey)
    return img

def get_F_beta(predicted, tgt, beta=0.3):
    predicted = predicted.mean(dim=1).clamp(0, 1)
    predicted = predicted.float().detach()
    predicted = torch.where(predicted > 0.5, 1.0, 0.0)

    tgt = tgt.mean(dim=1).clamp(0, 1)
    tgt = tgt.float().detach()
    tgt = torch.where(tgt > 0.5, 1.0, 0.0)

    TP = (torch.where((predicted-tgt) == 0.0, 1.0, 0.0) * torch.where(predicted == 1.0, 1.0, 0.0)).sum(dim=(1, 2))
    FP = (torch.where((predicted-tgt) != 0.0, 1.0, 0.0) * torch.where(predicted == 1.0, 1.0, 0.0)).sum(dim=(1, 2))
    FN = (torch.where((predicted-tgt) != 0.0, 1.0, 0.0) * torch.where(predicted == 0.0, 1.0, 0.0)).sum(dim=(1, 2))

    p = TP / (TP + FP + 1e-6)
    r = TP / (TP + FN + 1e-6)

    beta = 0.3
    f_measure = (beta + 1) * p * r / (beta * p + r + 1e-6)
    6
    # betas = torch.arange(0, 99) / 100
    # f_measures = []
    # for beta in betas:
    #     f_measure = (beta + 1) * p * r / (beta * p + r + 1e-6)
    #     f_measures.append(f_measure)
    # f_measures = torch.stack(f_measures, dim=1)
    # f_measure, _ = torch.max(f_measures, dim=-1)
    return f_measure.mean().cpu()


def get_MAE(predicted, tgt):
    predicted = predicted.mean(dim=1).clamp(0, 1)
    predicted = predicted.float().detach()

    tgt = tgt.mean(dim=1).clamp(0, 1)
    tgt = tgt.float().detach()

    mae = torch.mean(torch.abs(predicted-tgt), dim=(1, 2))
    return mae.mean().cpu()

def get_E_measure(predicted, tgt):
    predicted = predicted.mean(dim=1).clamp(0, 1)
    predicted = predicted.float().detach()

    tgt = tgt.mean(dim=1).clamp(0, 1)
    tgt = tgt.float().detach()

    tgt = torch.where(tgt > 0.5, 1.0, 0.0)
    predicted = torch.where(predicted > 0.5, 1.0, 0.0)

    Mgt = tgt - tgt.mean()
    Mf = predicted - predicted.mean()
    score_F = 2*Mgt*Mf / (Mgt**2 + Mf**2 + 1e-8)

    E_measure = (1 + score_F)**2 / 4
    return E_measure.mean().detach()


def get_center(mask):
    # print(mask.shape)
    assert len(mask.shape) == 2
    h, w = mask.shape
    x_grid = torch.arange(0, w).reshape(1, -1).repeat(h, 1).to(mask)
    y_grid = torch.arange(0, h).reshape(-1, 1).repeat(1, w).to(mask)
    x = (mask * x_grid).mean() / mask.mean()
    y = (mask * y_grid).mean() / mask.mean()
    return x.int(), y.int()


def split_to_piece(mask, x, y):
    h, w = mask.shape
    lt = mask[:y, :x]
    rt = mask[:y, x:]
    lb = mask[y:, :x]
    rb = mask[y:, x:]

    area = h*w
    w1 = x*y / area
    w2 = (w - x)*y / area
    w3 = x * (h - y) / area
    w4 = (w - x) * (h - y) / area
    return lt, rt, lb, rb, w1, w2, w3, w4


def ssim(mask1, mask2, eps=1e-8):
    h, w = mask1.shape
    n = h * w

    x = mask1.mean()
    y = mask2.mean()

    sigma_x = ((mask1 - x)**2).sum() / (n + eps)
    sigma_y = ((mask2 - y) ** 2).sum() / (n + eps)

    sigma_xy = ((mask1 - x) * (mask2 - y)).sum() / (n + eps)

    ssim = 2*x*y / (x**2 + y**2) * 2*sigma_x*sigma_y / (sigma_x**2+sigma_y**2) * 2*sigma_xy / (sigma_x*sigma_y)
    return ssim


def get_Sr(predicted, tgt):
    x, y = get_center(tgt)
    lt, rt, lb, rb, w1, w2, w3, w4 = split_to_piece(tgt, x, y)
    lt_, rt_, lb_, rb_, _, _, _, _ = split_to_piece(predicted, x, y)
    Sr = w1 * ssim(lt, lt_) + w2 * ssim(rt, rt_) + w3 * ssim(lb, lb_) + w4 * ssim(rb, rb_)
    return Sr

def get_So(p, tgt):
    return 1

def get_S_measure(predicted, tgt):
    predicted = predicted.mean(dim=1).clamp(0, 1)
    predicted = predicted.float().detach()

    tgt = tgt.mean(dim=1).clamp(0, 1)
    tgt = tgt.float().detach()

    tgt = torch.where(tgt > 0.5, 1.0, 0.0)
    predicted = torch.where(predicted > 0.5, 1.0, 0.0)

    b = tgt.shape[0]
    S_measures = []
    for i in range(b):
        S_measures.append(0.5 * get_Sr(predicted[i], tgt[i]) + 0.5 * 1)
    return torch.tensor(S_measures).mean().detach()

