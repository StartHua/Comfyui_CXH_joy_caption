import folder_paths
import os
import base64
import numpy as np
from PIL import Image,ImageOps, ImageFilter

import io

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

# D:\comfyui\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_CXH_ALY
# current_folder = os.path.dirname(os.path.abspath(__file__))

# 节点路径
def node_path(node_name):
    return os.path.join(custom_nodes_path,node_name)

# 创建文件夹
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径

# 获取所有图片文件路径
def get_all_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths


