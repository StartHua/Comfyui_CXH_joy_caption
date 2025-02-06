
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

classification_rules = """
You are a fashion image classifier. Analyze clothing images following these priority rules and categories. When an item could fit multiple categories, use the highest priority category.
Priority Order (Highest to Lowest):
1. MAN
2. WoMAN
Required Output Format:
[CATEGORY_NAME]

Classification Rules:
1. Always check categories in order from highest to lowest priority
2. Use the highest priority category that applies
3. Output only the category name in all caps
4. No additional text or explanations in output
"""

def process_category_name(category_name):
    # 如果字符串包含方括号，则删除它们
    if category_name.startswith('[') and category_name.endswith(']'):
        category_name = category_name[1:-1]
    return category_name

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
                "prompt":    ("STRING", {"multiline": True, "default": classification_rules},),
                "format": (["png", "jpg"],),
                "max_tokens":("INT", {"default": 1024, "min": 10, "max": 4048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = () #RETURN_TYPES = () RETURN_TYPES = ("DICT",)返回字典
    FUNCTION = "gen"
    OUTPUT_NODE = True #OUTPUT_NODE = True 没输出
    CATEGORY = "CXH/LLM"

    def gen(self,pipe,img_dir,save_dir,prompt,format,max_tokens,temperature,seed): 

        dir_files = batch_image(img_dir)

        # prompt = f"Determine whether the following pictures belong to the following types:{str(classifiy_type)},You only need to output the type, you do not need to output anything else to remember!"

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
 
            generated_text = process_category_name(res)

            if len(generated_text) >= 80:
                generated_text = "UNKNOWN"


            savePath = os.path.join(save_dir,generated_text)
             # 创建保存目录
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            lenName = str(index1)
            # img_file_name = f"{lenName}.{format}"
            img_file_name = os.path.basename(image_path)
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

        return ()

