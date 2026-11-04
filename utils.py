import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random


class Customdataset(Dataset):
    def __init__(self, transform=None,transform_seg=None, rgb_dataset=None, ir_dataset=None,label_dataset = None,Resize_H = None,Resize_W = None):
        self.image_rgb_paths = [f for f in os.listdir(rgb_dataset) if
                                os.path.isfile(os.path.join(rgb_dataset, f))]
        self.image_rgb_paths.sort()
        self.image_ir_paths = [f for f in os.listdir(ir_dataset) if
                               os.path.isfile(os.path.join(ir_dataset, f))]
        self.image_ir_paths.sort()
        self.image_label_paths = [f for f in os.listdir(label_dataset) if
                               os.path.isfile(os.path.join(label_dataset, f))]
        self.image_label_paths.sort()
        self.transform = transform
        self.transform_seg = transform_seg
        self.rgb_dataset = rgb_dataset
        self.ir_dataset = ir_dataset
        self.label_dataset = label_dataset
        self.resize_h = Resize_H
        self.resize_w = Resize_W

    def __getitem__(self, index):
        img1 = Image.open(self.rgb_dataset + self.image_rgb_paths[index]).convert('RGB')
        img2 = Image.open(self.ir_dataset + self.image_ir_paths[index]).convert('RGB')
        img3 = Image.open(self.label_dataset + self.image_label_paths[index])
        
        
        
        scale_options = [0.8,1.0,1.2,1.5,2]
        scale_factor = random.choice(scale_options)
        new_width = int(img1.width * scale_factor)
        new_height = int(img1.height * scale_factor)
        
        img1 = TF.resize(img1, (new_height, new_width), interpolation=InterpolationMode.BILINEAR)
        img2 = TF.resize(img2, (new_height, new_width), interpolation=InterpolationMode.BILINEAR)
        img3 = TF.resize(img3, (new_height, new_width), interpolation=InterpolationMode.NEAREST)
        
        h_flip = random.random() > 0.5
        
        if h_flip:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            img3 = TF.hflip(img3)
        
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(self.resize_h, self.resize_w))
        
        img1 = TF.crop(img1, i, j, h, w)
        img1 = self.transform(img1)
        img2 = TF.crop(img2, i, j, h, w)
        img2 = self.transform(img2)
        img3 = TF.crop(img3, i, j, h, w)
        img3 = self.transform_seg(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.image_rgb_paths) 


def get_test_images(paths, height=None, width=None):
    ImageToTensor = transforms.Compose([
        transforms.Resize((height,width)),
        transforms.ToTensor(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path)
        image_np = np.array(image, dtype=np.uint32)
        image = ImageToTensor(image).float().numpy()
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()

    return images


def get_image(path):
    image = Image.open(path).convert('RGB')

    return image

class Customdataset_val(Dataset):
    def __init__(self, transform=None,transform_seg=None, rgb_dataset=None, ir_dataset=None,label_dataset = None):
        self.image_rgb_paths = [f for f in os.listdir(rgb_dataset) if
                                os.path.isfile(os.path.join(rgb_dataset, f))]
        self.image_rgb_paths.sort()
        self.image_ir_paths = [f for f in os.listdir(ir_dataset) if
                               os.path.isfile(os.path.join(ir_dataset, f))]
        self.image_ir_paths.sort()
        self.image_label_paths = [f for f in os.listdir(label_dataset) if
                               os.path.isfile(os.path.join(label_dataset, f))]
        self.image_label_paths.sort()
        self.transform = transform
        self.transform_seg = transform_seg
        self.rgb_dataset = rgb_dataset
        self.ir_dataset = ir_dataset
        self.label_dataset = label_dataset

    def __getitem__(self, index):
        img1 = Image.open(self.rgb_dataset + self.image_rgb_paths[index]).convert('RGB')
        img2 = Image.open(self.ir_dataset + self.image_ir_paths[index]).convert('RGB')
        img3 = Image.open(self.label_dataset + self.image_label_paths[index])
        
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform_seg(img3)

        return img1, img2, img3
    def __len__(self):
        return len(self.image_rgb_paths) 
class SegmentationVisualizer:
    def __init__(self, segmentation_tensor):
        self.segmentation_tensor = segmentation_tensor

        self.color_map = np.array([
            [0, 0, 0],           # 0:unlabelled
            [179, 228, 228],     # 1:road
            [181, 57, 133],      # 2:sidewalk
            [67, 162, 177],      # 3:building
            [200, 178, 50],      # 4:lamp
            [132, 45, 199],      # 5:sign
            [66, 172, 84],       # 6:vegetation
            [179, 73, 79],       # 7:sky
            [76, 99, 166],       # 8:person
            [66, 121, 253],      # 9:car
            [137, 165, 91],      # 10:truck
            [155, 97, 152],      # 11:bus
            [105, 153, 140],     # 12:motorcycle
            [222, 215, 158],     # 13:bicycle
            [135, 113, 90]       # 14:pole
        ], dtype=np.uint8)

    def visualize(self):
        probabilities = torch.softmax(self.segmentation_tensor, dim=1)
        class_indices = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()  # shape (H, W)
        color_image = np.zeros((class_indices.shape[0], class_indices.shape[1], 3), dtype=np.uint8)
        for class_index in range(self.color_map.shape[0]):
            color_image[class_indices == class_index] = self.color_map[class_index]
        color_image_tensor = torch.tensor(color_image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)  # shape (1, 3, H, W)
        return color_image_tensor
