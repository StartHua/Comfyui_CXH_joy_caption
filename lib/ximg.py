# -*- encoding: utf-8 -*-
'''
@File         :ximg.py
@Description  :图片操作类封装
@Time         :2024/04/30 09:46:01
@Author       :ChenXingHua
@Version      :1.0
'''

import os
import torch
from PIL import Image, ImageOps, ImageSequence, ImageFile,UnidentifiedImageError
import numpy as np
import cv2 as cv
import io
import base64
import requests
from io import BytesIO

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2cv2(image:torch.Tensor) -> np.array:
    if image.dim() == 4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv.cvtColor(cv2image, cv.COLOR_RGB2BGR)

def cv22pil(cv2_img:np.ndarray) -> Image:
    cv2_img = cv.cvtColor(cv2_img, cv.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

# pil转io
def pil2iobyte(pil_image,format='PNG'):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format=format)  
    byte_arr = byte_arr.getvalue() 
    return byte_arr

# pil转64
def pilTobase64(pil_image,format='PNG'):
    byte_arr = pil2iobyte(pil_image,format)
    image_base64 = base64.b64encode(byte_arr).decode('utf-8') 
    return image_base64

def ioBytes2tensor(bytes):
    image = Image.open(bytes)
    return pil2tensor(image)

def getImageSize(image):
    if image.shape[0] > 0:
        image = torch.unsqueeze(image[0], 0)
    _image = tensor2pil(image)

    return (_image.width, _image.height)

# 转成mask
def imageToMask(img):
    i = img
    if i.mode == 'I':
        i = i.point(lambda i: i * (1 / 255))
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return tensor2pil(mask)

#   ret_masks.append(image2mask(_mask))
def image2mask(image:Image) -> torch.Tensor:
    _image = image.convert('RGBA')
    alpha = _image.split() [0]
    bg = Image.new("L", _image.size)
    _image = Image.merge('RGBA', (bg, bg, bg, alpha))
    ret_mask = torch.tensor([pil2tensor(_image)[0, :, :, 3].tolist()])
    return ret_mask

# 图像回帖
def croppImg(original_image,cropped_avatar,left_x,top_y):
    # 获取原始图像的大小
    original_width, original_height = original_image.size
    return croppImageBySize(cropped_avatar,left_x,top_y,original_width,original_height)

def croppImageBySize(cropped_avatar,left_x,top_y,original_w,original_h):
    # 获取原始图像的大小
    original_width, original_height = original_w,original_h
    # 获取头像的大小
    avatar_width, avatar_height = cropped_avatar.size
    # 创建一个与原始图像相同大小的透明图像
    extended_image = Image.new("RGBA", (original_width, original_height), (0, 0, 0, 0))
    # 将裁剪后的头像粘贴到新图像
    extended_image.paste(cropped_avatar, (left_x, top_y), cropped_avatar)
   
    return extended_image


# 将图片转换为Base64编码
def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 获取网络图片
def img_from_url(url):
    # 发送HTTP请求获取图片  
    response = requests.get(url)  
    response.raise_for_status()  # 如果请求失败，这会抛出异常
    # 将响应内容作为BytesIO对象打开，以便PIL可以读取它  
    image = Image.open(BytesIO(response.content))
    return image

def open_image(path):
    prev_value = None

    try:
        img = Image.open(path)
    except (UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(path)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
        return img