import glob
import os
from typing import Any, Optional, Callable

import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy
import torch
import torchvision.transforms
from PIL import Image

from albumentations import Compose, Resize, RandomCrop, HorizontalFlip, Normalize, RandomRotate90, Flip, Transpose, \
    OneOf, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, ShiftScaleRotate, OpticalDistortion, \
    GridDistortion, IAAPiecewiseAffine, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, HueSaturationValue, \
    Sharpen, PiecewiseAffine
from albumentations.pytorch import ToTensorV2
from imgaug.augmenters import Emboss
from torchvision import datasets
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, DatasetFolder
from torchvision.transforms import ToTensor



class AlbumentationsDataset(DatasetFolder):
    """
        处理数据增强跟上面的 TorchvisionDataset 的一致
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, idx):
        # label = self.labels[idx]
        # root = self.root[idx]
        """
               Args:
                   index (int): Index

               Returns:
                   tuple: (sample, target) where target is class_index of the target class.
               """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path, target = self.samples[idx]
        # print(path)
        sample = self.loader(path)
        sample = numpy.array(sample)
        if self.transform is not None:
            sample = self.transform(image=sample)
            # img = sample['image']
            # img = img.transpose(0,2)
            # plt.imshow(img)
            # plt.show()
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = sample['image']
        transformsTotensor = ToTensor()
        transformsToPLTImage = torchvision.transforms.ToPILImage()
        sample = transformsToPLTImage(sample)
        newsize=(224,224)
        sample= sample.resize(newsize)
        sample = transformsTotensor(sample)

        return sample, target

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

def val_aug(p=1):
    return A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.CenterCrop(height=128, width=128),
        A.CoarseDropout(p=0.4, max_holes=30, max_width=10, max_height=10),
        A.Normalize(mean=(0.4558, 0.4558, 0.4558), std=(0.2741, 0.2741, 0.2741)),
        ToTensorV2()

    ], p=p)
# def strong_aug(p=0.5):
#     return A.Compose([
#         A.SmallestMaxSize(max_size=160),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#         A.RandomCrop(height=128, width=128),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#     ])
def strong_aug(p=0.5):

    return A.Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(blur_limit=3,p=0.1),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            PiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            RandomBrightnessContrast(),
        ], p=0.2),
        A.HorizontalFlip(p=0.3),
        HueSaturationValue(p=0.1),
        A.Normalize(mean=(0.4558,0.4558,0.4558),std=(0.2741,0.2741,0.2741)),
        ToTensorV2()

    ], p=p)


if __name__ == "__main__":
    root = r'F:\__Earsenal\images\train\\'
    albumentations_dataset = AlbumentationsDataset(
        root=root,
        transform=strong_aug(),
    )
    loader = torch.utils.data.DataLoader(albumentations_dataset,
                                               batch_size=len(albumentations_dataset), shuffle=True,
                                               num_workers=4, pin_memory=False)
    data = next(iter(loader))
    a,b = data[0].mean(), data[0].std()
    print(a,b)
    # transform=strong_aug()
    # for root1, dirs, files in os.walk(root):
    #     for sDir in dirs=
    #         imgs_list = glob.glob(os.path.join(root1, sDir) + '/*.jpg')
    #         for file_path in imgs_list:
    #             image = cv2.imread(file_path)
    #
    #             # 默认OpenCV读取得到的是 BGR 图片
    #             # 转换 RGB 格式图片
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #             if transform:
    #                 image = transform(image=image)['image']
    #                 print(image)
    #                 plt.imshow(image)
    #                 plt.show()
    # for i in range(2):
    #     albumentations_dataset.__getitem__(i)
    #     #albumentations_dataset.collate_fn()
