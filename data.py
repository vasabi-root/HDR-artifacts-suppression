import cv2
import numpy as np
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from math import ceil

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from kornia.color import rgb_to_lab
from torchvision.io import decode_image

def plot_images(images):
    fig, axs = plt.subplots(ncols=len(images))
    for img, ax in zip(images, axs):
        ax.imshow(img)
        ax.set_axis_off()
    plt.show()

def calc_hists(img: cv2.Mat):
    assert len(img.shape) == 3 and img.shape[2] == 3
    bgr = cv2.split(img)
    return [cv2.calcHist(bgr, [i], None, [256], (0,256)) / bgr[0].size for i in range(3)]
    
def calc_cdfs(img: cv2.Mat):
    '''Calc the Cumulative Distribution Function through channels'''
    hists = calc_hists(img)
    return np.cumsum(hists, axis=1).squeeze()

def equalize_exposure(ref_cdfs: List[np.ndarray], img: cv2.Mat) -> cv2.Mat:
    cdfs = calc_cdfs(img)
    bgr = list(cv2.split(img))
    for i, (ref_cdf, cdf) in enumerate(zip(ref_cdfs, cdfs)):
        # intensity mapping func: ref_icdf( cdf[img] )
        imp = np.interp(cdf, ref_cdf, np.arange(256, dtype=np.int32))
        imp = imp.astype(np.uint8)
        bgr[i] = imp[bgr[i]]

        # fig, ax = plt.subplots(nrows=4)
        # ax[0].plot(ref_cdf, range(256))
        # ax[1].plot(cdf, range(256))
        # ax[2].plot(range(256), cdf)
        # ax[3].plot(range(256), imp)
        # plt.show(block=True)

    return cv2.merge(bgr)

def equalize_exposures(images, ref_idx, only_light=True, transform=None) -> List[cv2.Mat]:
    corrected = []

    ref = images[ref_idx]

    if only_light:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
        ref_cdfs = calc_cdfs(ref)
        for img in images:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            equalized = equalize_exposure(ref_cdfs, lab)
            l, a, b = cv2.split(equalized)
            corrected.append(cv2.merge([l, l, l]))
    else:
        ref_cdfs = calc_cdfs(ref)
        for img in images:
            equalized = equalize_exposure(ref_cdfs, img)
            corrected.append(equalized)
    if transform:
        transformed = []
        for img in corrected:
            tensor = torch.tensor(img).permute(2, 0, 1)
            transformed.append(transform(tensor))
    return corrected


def correct_gamma(img, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    corrected = cv2.LUT(img, lookUpTable)
    return corrected

def flow_to_rgb(flow: np.ndarray): # [2 H W] float
    assert flow.shape[0] == 2

    hsv = np.zeros((*flow.shape[1:], 3), np.uint8)
    mag, ang = flow[0], flow[1]
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
    

class RGBToYCbCr(object):
    def __call__(self, img):
        res = torch.stack((0. / 256. + img[0, :, :] * 0.299000 + img[1, :, :] * 0.587000 + img[2, :, :] * 0.114000,
                           128. / 256. - img[0, :, :] * 0.168736 - img[1, :, :] * 0.331264 + img[2, :, :] * 0.500000,
                           128. / 256. + img[0, :, :] * 0.500000 - img[1, :, :] * 0.418688 - img[2, :, :] * 0.081312))
        return  res
    
class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack((img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
                            img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 - (img[:, 2, :, :] - 128 / 256.) * 0.714136,
                            img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.) * 1.772),
                            dim=1)

IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tif', 'tiff']

class BracketingBunch:
    def __init__(self, bunch_dir: Path, num_exposures: int=None, ref_idx: int=None, is_cv=False):
        self.dir = Path(bunch_dir)
        self.num_exposures = num_exposures
        # self.inference_size = (np.array([4080, 3060]) / 4).astype(np.int32) # album [W H] [1536, 1152] / 4
        # self.inference_shape = [self.inference_size[1], self.inference_size[0], 3]
        self.is_cv = is_cv
        if is_cv:
            self._evaluate_images_cv()
        else:
            self._evaluate_images()
        self._sort_by_median()
        self._remove_extra_exposures()
        
        assert len(self) == self.num_exposures
        assert ref_idx == None or 0 <= ref_idx < len(self.images)
        self.ref_idx = ref_idx if ref_idx else len(self.images) // 2

    def get_median(self):
        return self.images

    def equalize_ref(self, only_light=True) -> List[cv2.Mat]:
        corrected = []

        ref = self.images[self.ref_idx]

        if only_light:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
            for img in self.images:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                cdfs = calc_cdfs(lab)
                equalized = equalize_exposure(cdfs, ref)
                l, a, b = cv2.split(equalized)
                corrected.append(cv2.merge([l, l, l]))
        else:
            for img in self.images:
                cdfs = calc_cdfs(img)
                equalized = equalize_exposure(cdfs, ref)
                corrected.append(equalized)
        return corrected

    
    def equalize_exposures(self, only_light=True, to_tensor=False, transform=None) -> List[cv2.Mat]:
        corrected = []

        ref = self.images[self.ref_idx]
        if transform:
            to_tensor = True

        if only_light:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
            ref_cdfs = calc_cdfs(ref)
            for img in self.images:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                equalized = equalize_exposure(ref_cdfs, lab)
                l, a, b = cv2.split(equalized)
                corrected.append(cv2.merge([l, l, l]))
        else:
            ref_cdfs = calc_cdfs(ref)
            for img in self.images:
                equalized = equalize_exposure(ref_cdfs, img)
                corrected.append(equalized)

        if to_tensor:
            for i in range(self.num_exposures):
                corrected[i] = torch.tensor(corrected[i]).permute(2, 0, 1)
            corrected = torch.stack(corrected)
        
        if transform:
            transformed = []
            for img in corrected:
                transformed.append(transform(img))
            corrected = torch.stack(transformed)

        return corrected
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    
    def _evaluate_images_cv(self) -> None:
        self.images = []
        self.names = [name for name in os.listdir(self.dir) if name.lower().split('.')[-1] in IMG_EXTENSIONS]

        for img_name in self.names:
            img = cv2.imread(self.dir / img_name)
            self.images.append(img)

        self.inference_shape = self.images[0].shape[1:]
    
    def _evaluate_images(self) -> None:
        self.images = []
        self.names = os.listdir(self.dir)

        for img_name in self.names:
            img = decode_image(self.dir / img_name)
            self.images.append(img)

        self.inference_shape = self.images[0].shape[:2]


    def _sort_by_median(self):
        zipped = sorted(zip(self.images, self.names), key=lambda x: np.median(x[0]))
        self.images, self.names = zip(*zipped)

    def _remove_extra_exposures(self):
        if self.num_exposures:
            stride = ceil(len(self.images) / self.num_exposures)
            images = [self.images[i] for i in range(0, (self.num_exposures-1)*stride, stride)]
            images.append(self.images[-1])
            self.images = images
        else:
            self.num_exposures = len(self.images)

    def to_tensor(self, transform: v2.Transform=None):
        try:
            images = self.images
            if self.is_cv:
                images = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)) for img in self.images]
            if transform:
                transformed = []
                for img in images:
                    transformed.append(transform(img))
                images = torch.stack(transformed)
            else:
                images = torch.stack(images)
        except Exception as e:
            pass

        return images

class BracketingBunchWeighted(BracketingBunch):
    def __init__(self, bunch_dir, weights_dir, num_exposures = None, ref_idx = None, is_cv=False):
        super().__init__(bunch_dir, num_exposures, ref_idx, is_cv)
        self.weights = [decode_image(weights_dir / name) for name in os.listdir(weights_dir)]
        assert len(self.weights) == num_exposures
    
    def weights_to_tensor(self, transform: v2.Transform=None):
        try:
            images = self.weights
            if transform:
                transformed = []
                for img in images:
                    transformed.append(transform(img))
                images = torch.stack(transformed)
            else:
                images = torch.stack(images)
        except Exception as e:
            pass

        return images


class BunchDataset(Dataset):
    def __init__(self, root_dir: Path, num_exposures: int, transform_lr: v2.Transform=None, transform_hr=None):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.num_exposures = num_exposures
        self.bunch_names = [name for name in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / name)]
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def init_bunch_names(self):
        self.bunch_names = []
        for name in os.listdir(self.root_dir):
            bunch_dir = self.root_dir / name
            if os.path.isdir(bunch_dir) and os.listdir(bunch_dir) >= self.num_exposures:
                self.bunch_names.append(name)

    def __len__(self):
        return len(self.bunch_names)
    
    def __getitem__(self, idx):
        bunch = BracketingBunch(self.root_dir / self.bunch_names[idx], self.num_exposures)
        
        exposures_lr = bunch.to_tensor(self.transform_lr)
        exposures_hr = bunch.to_tensor(self.transform_hr)

        return self.bunch_names[idx], exposures_lr, exposures_hr
    
class BunchWeightedDataset(BunchDataset):
    def __init__(self, root_dir, weights_dir, num_exposures, transform_exposure = None, transform_w=None):
        super().__init__(root_dir, num_exposures)
        self.transform_exposure = transform_exposure
        self.weights_dir = Path(weights_dir)
        self.transform_w = transform_w
        self.weight_names = os.listdir(weights_dir)

        self.names = []
        for name in self.bunch_names:
            if name in self.weight_names:
                self.names.append(name)
        # assert len(self.weight_names) == len(self.bunch_names)
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        bunch = BracketingBunchWeighted(self.root_dir / name, self.weights_dir / name, self.num_exposures, is_cv=True)
        
        exposures = bunch.to_tensor(transform=self.transform_exposure)
        equalized = bunch.equalize_exposures(only_light=True, to_tensor=True, transform=self.transform_w)[:, 0].unsqueeze(1)
        weights = bunch.weights_to_tensor(self.transform_w)[:, 0].unsqueeze(1)

        return name, exposures, equalized, weights
    
    
if __name__ == '__main__':
    bunch = BracketingBunch(r'D:\windows\Documens\Diploma\dataset\old\aligned\04', num_exposures=3)

    # transform = v2.Compose([
    #     # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    #     v2.Resize(720),
    #     RGBToYCbCr(),
    # ])
    # dataset = BunchDataset(r'D:\windows\Documens\Diploma\dataset\old\aligned', num_exposures=3, transform_lr=transform)
    # print(len(dataset))
    # plot_images(dataset[0][1].permute((0, 2, 3, 1)))

    # dataset = BunchWeightedDataset(
    #     r'D:\windows\Documens\Diploma\dataset\train\train', 
    #     r'D:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_0_ep\train\weights', 
    #     num_exposures=3
    # )

    # print(len(dataset))
    # plot_images(dataset[0][3].permute((0, 2, 3, 1)))

    from utils.aligning import align_pyramids

    dataset_dir = Path(r'D:\windows\Documens\Diploma\dataset\custom')
    aligned_dir = dataset_dir.with_name(dataset_dir.stem+'_aligned')
    for name in os.listdir(dataset_dir):
        bunch_dir = dataset_dir / name
        aligned_bunch = aligned_dir / name
        if os.path.isdir(bunch_dir):
            bunch = BracketingBunch(bunch_dir, is_cv=True)
            images = bunch.images
            images = align_pyramids(
                images,
                [str(i)+'.jpg' for i in range(len(images))],
                aligned_bunch,
                depth=1,
            )

            # for i, img in enumerate(bunch):
            #     cv2.imwrite(bunch_dir / f'{i}.jpg', img)

    # dataset = BunchDataset(
    #     r'D:\windows\Documens\Diploma\dataset\kalantari\Training', 
    #     r'D:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_0_ep\train\weights',
    # )