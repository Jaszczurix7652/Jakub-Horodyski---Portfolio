import glob
import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import json
from utils.diffusion_utils import load_latents

class MS_COCODataset(Dataset):
    def __init__(self, split, im_path, annotations_path, im_size=256, im_channels=3, use_latents=False, latent_path=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_path = im_path
        self.annotations_path = annotations_path
        self.latent_maps = None
        self.use_latents = use_latents
        
        self.images, self.captions = self.load_images_and_captions()
        
        if use_latents and latent_path is not None:
            self.latent_maps = load_latents(latent_path)
            if len(self.latent_maps) != len(self.images):
                print('Mismatch between latent maps and images count')
                self.use_latents = False
    
    def load_images_and_captions(self):
        with open(self.annotations_path, 'r') as f:
            annotations = json.load(f)
        
        ims = []
        captions = []
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            image_info = next((item for item in annotations['images'] if item["id"] == image_id), None)
            if image_info:
                file_name = image_info['file_name']
                ims.append(os.path.join(self.im_path, file_name))
                captions.append(caption)
        
        print(f'Found {len(ims)} images and captions')
        return ims, captions

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.use_latents:
            latent = self.latent_maps[index]
            return latent, self.captions[index]
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.im_size, self.im_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])(im)
            im.close()
            return im_tensor, self.captions[index]

