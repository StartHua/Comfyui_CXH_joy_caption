
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import folder_paths
import time

from .lib.ximg import *
from .lib.xmodel import *

class CXH_Min2_6_classifiy :

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_Hg_Pipe",),
                "img_dir": ("STRING", {"multiline": False, "default": ""},),
                "save_dir":   ("STRING", {"multiline": False, "default": ""},),
                "classifiy_type":   ("STRING", {"multiline": True, "default": "man,woman"},),
                "format": (["png", "jpg"],),
                "max_tokens":("INT", {"default": 1024, "min": 10, "max": 4048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,pipe,img_dir,save_dir,classifiy_type,format,max_tokens,temperature,seed): 

        dir_files = batch_image(img_dir)

        prompt = f"Determine whether the following pictures belong to the following types:{str(classifiy_type)},You only need to output the type, you do not need to output anything else to remember!"

         # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        index1 = 0
        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            start = time.time()   
            input_image = open_image(image_path)
            input_image = ImageOps.exif_transpose(input_image)
            image = input_image.convert("RGB") 

            question = prompt
            msgs = [{'role': 'user', 'content': [image, question]}]

            res = pipe.text_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=pipe.tokenizer
            )

            ## if you want to use streaming, please make sure sampling=True and stream=True
            ## the model.chat will return a generator
            res = pipe.text_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=pipe.tokenizer,
                sampling=False,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            generated_text = ""
            for new_text in res:
                generated_text += new_text
                print(new_text, flush=True, end='')

            savePath = os.path.join(save_dir,generated_text)
             # 创建保存目录
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            lenName = str(index1)
            img_file_name = f"{lenName}.{format}"
            input_image = image
            if format != "png":
                if input_image.mode == "RGBA":
                    input_image = input_image.convert("RGB")
                    
            img_save_path = os.path.join(savePath, img_file_name)
            input_image.save(img_save_path)

            end = time.time()
            execution_time = calculate_seconds_difference(start, end)
            temp = f":{execution_time:.3f}s"
            index1 = index1 + 1
            print(str(index1)+"/"+str(len(dir_files)) +":"+temp)    

        return (str(index1),)

