import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import random
import numpy as np

def cv_random_flip(img, gt):
    if random.randint(0, 1):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, gt

def randomCrop(image, gt):
    border = 30
    image_width, image_height = image.size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return image.crop(region), gt.crop(region)

def randomRotation(image, gt):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        gt = gt.rotate(angle, Image.BICUBIC)
    return image, gt

def colorEnhance(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.0, 2.0))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.0, 3.0))
    return image

def randomPeper(gt):
    gt = np.array(gt)
    noiseNum = int(0.0015 * gt.shape[0] * gt.shape[1])
    for _ in range(noiseNum):
        x = random.randint(0, gt.shape[0] - 1)
        y = random.randint(0, gt.shape[1] - 1)
        gt[x, y] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(gt)

class CamImgTrain(data.Dataset):
    def __init__(self, image_root, gt_root, image_size):
        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.gts = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.filter_files()
        self.image_size = image_size

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image, gt = cv_random_flip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)

        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

def get_loader(image_root, gt_root, batch_size, image_size, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CamImgTrain(image_root, gt_root, image_size)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def split_dataset(image_root, gt_root, image_size=352, val_ratio=0.1):
    from sklearn.model_selection import train_test_split

    images = sorted([
        os.path.join(image_root, f)
        for f in os.listdir(image_root)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    gts = sorted([
        os.path.join(gt_root, f)
        for f in os.listdir(gt_root)
        if f.endswith('.jpg') or f.endswith('.png')
    ])

    paired = list(zip(images, gts))
    paired = [p for p in paired if Image.open(p[0]).size == Image.open(p[1]).size]
    train_paired, val_paired = train_test_split(paired, test_size=val_ratio, random_state=42)

    class CustomDataset(data.Dataset):
        def __init__(self, paired_paths):
            self.paired = paired_paths
            self.img_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

        def __len__(self):
            return len(self.paired)

        def __getitem__(self, index):
            img_path, gt_path = self.paired[index]
            image = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
            return image, gt

    return CustomDataset(train_paired), CustomDataset(val_paired)
