import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2

def initialize_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

def make_dir(path, parents=False):
    Path(os.path.expanduser(path)).mkdir(parents=parents, exist_ok=True)

def plot_one_sample_from_images(images, plot_path, filename, isRange01=False):
    ''' Plot images
    Args: 
        images: (c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
        isRange01: True/False, Normalization will be different. 
    '''
    if isRange01:
        images = images
    else:
        max_pix = torch.max(torch.abs(images))
        if max_pix != 0.0:
            images = ((images/max_pix) + 1.0)/2.0
        else:
            images = (images + 1.0) / 2.0
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)

    images = np.swapaxes(np.swapaxes(torch.squeeze(images).cpu().numpy(), 0, 1), 1, 2)
    idx=0
    plt.imsave(os.path.join(plot_path, filename), images)


def mark_point(imgs, fixs, ds=7, isRed=True):
    '''
    Mark a point in the given image.
    Args:
        imgs: (b, 3, h, w), tensor, any range
        fixs: (b, 2), (float x, float y), tensor, -1~1
    return:
        img_marked: (b, 3, h, w)
    '''
    img_s = imgs.size()

    fixs = (fixs + 1)/2.0 # 0~1
    fixs[:,0] = fixs[:,0] * img_s[-1]
    fixs[:,1] = fixs[:,1] * img_s[-2]
    fixs = fixs.to(torch.int)

    for b in range(img_s[0]):
        if isRed:
            imgs[b, :, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 0.0
            imgs[b, 0, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 2.0
        else:
            imgs[b, :, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 0.0
            imgs[b, 2, fixs[b,1]-ds:fixs[b,1]+ds, fixs[b,0]-ds:fixs[b,0]+ds] = 2.0
    return imgs


def mark_fixations(imgs, fixs, ds=7, isRed=True):
    '''
    Mark fixation points in the given images. This function is used to mark a fixation.
    Args:
        imgs: (b, 3, h, w), tensor, any range
        fixs: (b, 2), (float x, float y), tensor, -1~1
    return:
        img_marked: (b, 3, h, w)
    '''

    imgs = normalize_min_max(imgs)
    imgs = mark_point(imgs, fixs, ds=ds, isRed=isRed)

    return (imgs -0.5)*2.0


def mark_fixations_history(imgs, fixs_h, ds=21, isLastRed=True):
    '''
    Mark fixation history in the given images. This function is used to mark fixation history.
    Args:
        imgs: (b, 3, h, w), tensor, any range
        fixs: (b, step, 2), (float x, float y), tensor, -1~1
    return:
        img_marked: (b, 3, h, w)
    '''
    n_steps = fixs_h.size(1)
    imgs = normalize_min_max(imgs)
    img_m = imgs
    for step in range(n_steps):
        if step == n_steps-1:
            imgs = mark_point(imgs, fixs_h[:, step, :], ds=ds, isRed=isLastRed)
        else:
            imgs = mark_point(imgs, fixs_h[:, step, :], ds=ds, isRed=False)

    return (imgs -0.5)*2.0

def add_heatmap_on_image(heatmap, image):
    '''Visualize heatmap on image. This function is not based on batch.
    Args:
        heatmap: (h, w), 0~1 ranged numpy array
        image: (3, h, w), 0~1 ranged numpy array
        heatmap and image must be in the same size. 
    return:
        hm_img: (h, w, 3), 0~255 ranged numy array
    '''
    heatmap_cv = heatmap * 255
    heatmap_cv = cv2.applyColorMap(heatmap_cv.astype(np.uint8), cv2.COLORMAP_JET) #(h, w, 3)
    heatmap_cv = cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB)
    image_cv = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)*255
    hm_img = cv2.addWeighted(heatmap_cv, 0.7, image_cv.astype(np.uint8), 0.3, 0)

    return hm_img

def add_heatmap_on_image_tensor(heatmap, image, resize_s=(112,112), isNormHM=True, device='cpu'):
    '''
    Visualize heatmap on image. This function works based on batched tensors
    Args:
        heatmap: (b, h, w), any ranged tensor
        image: (b, 3, h, w), any ranged tensor
        resize_s: (int, int), heatmap and image will be resized to this size
        isNormHM: True/False, if True, heatmap will be normalized to 0~1
    return:
        hm_img: (b, 3, h, w), 0~1 ranged tensor
    '''
    ret = []
    bs = image.size(0)

    heatmap = heatmap.unsqueeze(1) #(b, 1, h, w)
    if resize_s is not None:
        heatmap = torch.nn.functional.interpolate(heatmap, resize_s, mode='bilinear')
        image = torch.nn.functional.interpolate(image, resize_s, mode='bilinear')

    if isNormHM:
        heatmap = normalize_min_max(heatmap)
    image = normalize_min_max(image)

    for b in range(bs):
        hm_i = torch.squeeze(heatmap[b]).cpu().numpy()
        image_i =image[b].cpu().numpy()
        hmimg = add_heatmap_on_image(hm_i, image_i)
        ret.append(hmimg)
    ret = np.stack(ret, axis=0) # 0~255 ranged numpy array (b, h, w, 3)
    ret = np.swapaxes(np.swapaxes(ret.astype(np.float32), 2, 3), 1, 2) # 0~255 ranged numpy array(b, 3, h, w)
    ret = torch.tensor(ret/255.0, device=device)

    return ret

class Load_PIL_to_tensor():
    '''
    Modified 2022.11.02, Minkyu
        1) mean and std can be given as inputs. 
    '''
    def __init__(self, img_s_load=227, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], isResize=True):
        if isResize:
            self.transform = transforms.Compose([
                transforms.Resize(img_s_load),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ])

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        # https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py#L244
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def load_image_to_tensor(self, path: str) -> torch.Tensor:
        image_pil = self.pil_loader(path)
        #image_tensor = transforms.ToTensor()(image_pil).unsqueeze_(0)
        image_tensor = self.transform(image_pil)
        #print(image_tensor.size(), torch.max(image_tensor), torch.min(image_tensor))
        
        return image_tensor.unsqueeze(0)

def normalize_min_max(fms, isStable=False):
    ''' Normalize input fms range from 0 to 1.

    Args:
        fms: (b, c, h, w)
    return:
        fms_norm: (b, c, h, w)
    '''
    fms_s = fms.size()
    if len(fms_s) == 3:
        fms = fms.unsqueeze(1)
        fms_s = fms.size()

    min_val = torch.min(fms.reshape(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    max_val = torch.max(fms.reshape(fms_s[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    max_min = max_val - min_val

    fms_norm = (fms - min_val) / max_min
    fms_norm[fms_norm != fms_norm] = 0
    assert not torch.isnan(fms_norm).any(), f"[Assert] from noralize_min_max in misc, nan is detected. "
    return fms_norm

def scale_image(images):
    max_pix = torch.max(torch.abs(images))
    if max_pix != 0.0:
        images = ((images/max_pix) + 1.0)/2.0
    else:
        images = (images + 1.0) / 2.0
    return images

def concatenate_images_horizontally(img1, img2, img3, target_size, margin_width):
    # Transform to resize the image
    resize_transform = transforms.Resize(target_size)

    img1 = scale_image(img1)
    img2 = scale_image(img2)
    img3 = scale_image(img3)

    # Ensure images are in the format (3, h, w) and resize them
    img1 = resize_transform(img1.squeeze(0) if img1.dim() == 4 else img1)
    img2 = resize_transform(img2.squeeze(0) if img2.dim() == 4 else img2)
    img3 = resize_transform(img3.squeeze(0) if img3.dim() == 4 else img3)

    # Create a white margin (zero values for white in RGB)
    _, h, _ = img1.size()
    margin = torch.zeros(3, h, margin_width, device='cuda')

    # Concatenate images and margins
    concatenated_images = torch.cat([img1, margin, img2, margin.clone(), img3], dim=2)

    return concatenated_images
