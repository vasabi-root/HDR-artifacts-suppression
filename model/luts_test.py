
import torch
import torch.nn as nn
import torch.functional as F
from data import RGBToYCbCr, YCbCrToRGB
# from torch.utils.data import DataLoader
from torchvision.transforms import v2
import numpy as np
import cv2
from pathlib import Path
import os
import signal

from data import BracketingBunch
from utils.plot import plot_bunches, plot_imgs, plot_luts

from efficientvit.models.nn import EfficientViTBlock, ResidualBlock, MBConv, IdentityLayer
from efficientvit.models.efficientvit import EfficientViTSam
from model.efficient_vit import lightMSLA
from torchsummary import summary
from model.gfu.guided_filter import FastGuidedFilter
from losses.gmef_ssim import clamp_min
from utils.data import save_image_tensor


def eval_luts(luts_dir: Path):
    luts = []
    for name in os.listdir(luts_dir):
        with open(luts_dir / name, 'r') as f:
            str_vals = f.readline().rstrip().split(',')[:-1]
            lut = list(map(float, str_vals))
            luts.append(lut)

    luts = np.array(luts).clip(0.0, 1.0)
    return luts

def get_weight_map(luts_dir: Path):
    num = len(os.listdir(luts_dir))
    fs = [open(os.path.join(luts_dir, str(i) + "_weight.txt"), 'r') for i in range(num)]
    data = [f.read().split(',') for f in fs]
    weight_map = np.zeros([num, 256], np.double)
    for k in range(num):
        for j in range(256):
            d = float(data[k][j])
            if d > 1.0:
                d = 1.0
            if d < 0.0:
                d = 0.0
            weight_map[k][j] = d
    [f.close() for f in fs]
    return weight_map

def get_fusion_mask(images, weight_map):
    K, C, H, W = images.shape
    weights = np.zeros((K, 1, H, W))
    for k in range(K):
        weights[k:k+1] = weight_map[k:k+1][:, images[k:k+1]]
    return torch.from_numpy(weights)

def get_gauss_weight(
        img: np.array, # изображение, для которого считается весовая функция
        bound: float,  # средняя медиана по всем изображениям
        sigma: float   # среднее СКО по всем изображениям
    ) -> np.ndarray:
        '''
        Расчитывает весовую функцию на основе функции Гаусса.
        Возвращает массив весов для 8-битного диапазона от 0 до 255.
        
        - `img` -- изображение, для которого нужно посчитать вес
        - `bound` -- средняя медиана по всем изображениям сцены
        - `sigma` -- СКО функции Гаусса
        
        Матожидание функции Гаусса вычисляется как 
        128 - (`bound` - <медиана  `img`>)
        '''
        med = np.median(img)
        diff = bound - med
        num = 3*256
        weight = signal.gaussian(num, std=sigma, sym=False)

        low = int(num*1/3 - diff)
        up  = int(num*2/3 - diff)

        weight = weight[low:up]
        return weight

def apply_luts_Y(bunch: torch.Tensor, luts: torch.Tensor):
    assert len(luts) == len(bunch)
    weights = []
    for Y, lut in zip(bunch, luts):
        if isinstance(Y, torch.Tensor):
            Y, lut = Y.cpu().detach().numpy(), lut.cpu().detach().numpy()
        if Y.dtype != np.uint8:
            Y = (Y * 255).astype(np.uint8)
        weight = lut[Y]
        weights.append(weight)
    return np.array(weights)

def normalize(arr, t_min=0.0, t_max=1.0):
    diff = t_max - t_min
    diff_arr = (arr.max() - arr.min()).clip(1e-6)

    normed = (arr - arr.min())*diff / diff_arr
    return normed

def apply_weights_YCbCr_torch(bunch, weights):
    # bunch_np = bunch.cpu().detach().numpy()
    # Ys, Cb, Cr = bunch.permute(1, 0, 2, 3)
    Ys = bunch[:, 0]
    Cb = bunch[:, 1]
    Cr = bunch[:, 2]

    Y_f = (Ys * weights).sum(0) / weights.sum(0).clamp(1e-6)

    Wb = (torch.abs(Cb - 0.5) / (torch.sum(torch.abs(Cb - 0.5), dim=0)).clamp(1e-6)).clamp(0, 1)
    Wr = (torch.abs(Cr - 0.5) / (torch.sum(torch.abs(Cr - 0.5), dim=0)).clamp(1e-6)).clamp(0, 1)
    Cb_f = torch.sum(Wb * Cb, dim=0).clamp(0, 1)
    Cr_f = torch.sum(Wr * Cr, dim=0).clamp(0, 1)

    fused = torch.stack([Y_f, Cb_f, Cr_f]).unsqueeze(0)

    return fused

def apply_weights_YCbCr_cv(bunch, weights):
    if isinstance(weights, torch.Tensor):
        bunch = bunch.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
    # Ys, Cb, Cr = bunch.permute(1, 0, 2, 3)
    Ys = bunch[:, :, :, 0]
    Cb = bunch[:, :, :, 1]
    Cr = bunch[:, :, :, 2]


    Y_f = (Ys * weights).sum(0) /  weights.sum(0).clip(1e-6)

    Wb = (np.abs(Cb - 0.5) / (np.sum(np.abs(Cb - 0.5), 0)).clip(1e-6)).clip(0, 1)
    Wr = (np.abs(Cr - 0.5) / (np.sum(np.abs(Cr - 0.5), 0)).clip(1e-6)).clip(0, 1)
    Cb_f = np.sum(Wb * Cb, 0)
    Cr_f = np.sum(Wr * Cr, 0)

    fused = cv2.merge([Y_f, Cr_f, Cb_f]).clip(0, 255).astype(np.uint8)

    return fused

def test_merge_mefssim():
    # bunch_dir = Path(r'D:\windows\Documens\Diploma\dataset\old\aligned\04')
    bunch_dir = Path(r'D:\windows\Documens\Diploma\results\sequences\105')
    bunch = BracketingBunch(bunch_dir, 3, is_cv=True) # 76
    res_dir = Path('results/test_mefssim')
    os.makedirs(res_dir, exist_ok=True)
    
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(bunch.images) * 255

    # merge_robertson = cv2.createMergeRobertson()
    # res_robertson = merge_robertson.process(bunch.images, np.array([1.0, 4.0, 16.0])) * 255

    cv2.imwrite(res_dir / (bunch_dir.stem+'_mertens.jpg'), res_mertens.clip(0, 255).astype(np.uint8))
    # cv2.imwrite(res_dir / (bunch_dir.stem+'_robertson.jpg'), res_robertson.clip(0, 255).astype(np.uint8))


def test_luts():
    plots_dir = Path('plots')
    transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        # v2.Resize(700),
        RGBToYCbCr(),
    ])
    to_rgb = YCbCrToRGB()
    # bunch_dir = Path(r'D:\windows\Documens\Diploma\dataset\old\aligned\04')
    bunch_dir = Path(r'D:\windows\Documens\Diploma\results\sequences\17')
    bunch = BracketingBunch(bunch_dir, 3) # 76

    tensors = bunch.to_tensor(transform)
    Ys = tensors[:, 0].unsqueeze(1)
    Cb = tensors[:, 1].unsqueeze(1)
    Cr = tensors[:, 2].unsqueeze(1)
    Wb = (torch.abs(Cb - 0.5)) / torch.sum(torch.abs(Cb - 0.5).clamp(1e-6), dim=0)
    Wr = (torch.abs(Cr - 0.5)) / torch.sum(torch.abs(Cr - 0.5).clamp(1e-6), dim=0)
    Cb_f = torch.sum(Wb * Cb, dim=0, keepdim=True).clamp(0, 1)
    Cr_f = torch.sum(Wr * Cr, dim=0, keepdim=True).clamp(0, 1)

    # luts_dir = Path(r'luts\3')
    luts_dir = Path(r'D:\windows\Documens\Diploma\results\luts\train\3_1_scale_mef_ssim_lum_5_ep')
    luts = eval_luts(luts_dir)
    luts_meflut = get_weight_map(luts_dir)

    # Y_masks = get_fusion_mask(np.expand_dims(Ys*255, 1).astype(np.uint8), torch.from_numpy(luts))
    Y_masks = get_fusion_mask((Ys*255).byte(), luts)
    plot_bunches(Y_masks)

    EPS=1e-6
    sum_weights = torch.sum(Y_masks + EPS, dim=0)
    sum_weights = torch.clamp(sum_weights, min=EPS)
    Y_masks = (Y_masks + EPS) / sum_weights

    # Y_f = torch.sum(Y_masks * Ys, dim=0, keepdim=True).clamp(0, 1)
    Y_f = ((Y_masks * Ys).sum(0) / Y_masks.sum(0).clamp(EPS)).clamp(0, 1).unsqueeze(0)
    fused_rgb = YCbCrToRGB()(torch.cat((Y_f, Cb_f, Cr_f), dim=1))

    save_dir = Path('plots')
    os.makedirs(save_dir, exist_ok=True)
    save_image_tensor(fused_rgb, save_dir / 'fused_17.jpg')


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model = PairEVitHDR(in_channels=3, num_exposures=3, embed_dim=32).to(device)
    # sample_input_lr = torch.randn(1, 3, 3, 256, 356).to(device)
    # sample_input_hr = torch.randn(1, 3, 3, 1500, 2000).to(device)
    # output, _ = model(sample_input_lr, sample_input_hr)
    # # print(output.shape)
    # summary(model, [sample_input_lr.shape[1:],sample_input_hr.shape[1:]], 1)

    # test_merge_mefssim()
    test_luts()
    if False:
        plots_dir = Path('plots')
        transform = v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            # v2.Resize(700),
            # RGBToYCbCr(),
        ])
        to_rgb = YCbCrToRGB()
        bunch = BracketingBunch(Path(r'D:\windows\Documens\Diploma\dataset\old\aligned\04'), 3) # 76

        # luts_dir = Path(r'D:\windows\Documens\Diploma\results\luts\train\3frames_1_scale_grad_1_ep_ssim')
        luts_dir = Path(r'luts\3')
        luts_dir = Path(r'D:\windows\Documens\Diploma\results\luts\train\3_1_scale_mef_ssim_lum_5_ep')
        luts = eval_luts(luts_dir)
        luts_meflut = get_weight_map(luts_dir)
        # plot_luts(torch.tensor(luts), plots_dir / 'luts.png')
        # plot_luts(torch.tensor(luts_meflut), plots_dir / 'luts_.png')
        
        arrays = bunch.images
        masks = get_fusion_mask(np.expand_dims(bunch.images.mean(1), 1).astype(np.uint8), luts)
        plot_bunches(masks)
        bgrs = [array.transpose(1, 2, 0) for array in arrays]
        yuvs = np.array([cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb) for bgr in bgrs])

        weights = apply_luts_Y(yuvs[:, :, :, 0], luts)
        # fused_rgb = to_rgb(apply_weights_YCbCr_torch(bunch.to_tensor(transform), torch.tensor(weights))).clamp(0, 1)
        yuv = apply_weights_YCbCr_cv(yuvs, weights)
        fused_bgr = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)



        # plot_bunches(bunch.to_tensor())

        # fused = ((weights * bunch.to_tensor(transform)).sum(0) / weights.sum(0).clamp(1e-6)).clamp(1e-6, 1.0)
        # fused = bunch.to_tensor(transform).sum(0) / 3

        # plot_imgs(fused_rgb, save_name=plots_dir / 'fused.png')
        plot_luts(torch.tensor(luts), plots_dir / 'luts.png')

        cv2.imwrite(plots_dir / 'fused.jpg', fused_bgr)

    




