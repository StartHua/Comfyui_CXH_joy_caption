import torch
import torchvision.transforms.functional as F
import io
import os
from typing import List
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFont
import random
import numpy as np
import re
import time
from .lib.ximg import *
from .lib.xmodel import *
from comfy.utils import ProgressBar, common_upscale
import torchvision.transforms.functional as TVF

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    # imports.remove("flash_attn")
    return imports


import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))



class CXH_IC_Lora_Florence2Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tip_pipe":   ("STRING", {"multiline": False, "default": "", "forceInput": True},),
                "florence2_model": ("FL2MODEL", ),
                "format": (["png", "jpg"],),
                "max_new_tokens":("INT", {"default": 512, "min": 10, "max": 4096, "step": 1}),
                "dir1": ("STRING", {"default": ""}),
                "dir2":  ("STRING", {"default": ""}),
                "saveDir":  ("STRING", {"default": ""}),
                "text1": ("STRING", {"default": "", "multiline": True, "label": "Text Box 1"}),
                "text2": ("STRING", {"default": "", "multiline": True, "label": "Text Box 2"}),
                "template": ("STRING", {"default": "Realistic style, [cloth-on], the image pair highlights a transformation from a clothing sample photo to the effect of actually wearing it. [image1] {caption} [image2] a female model is wearing the cloth from [image1] with {caption}", "multiline": True, "label": ""}),
                "direction": (
                [   'right',
                    'down',
                    'left',
                    'up',
                ],
                {
                "default": 'right'
                }),
                "match_image_size": ("BOOLEAN", {"default": True}),
                }
            }
    
    RETURN_TYPES = ( "STRING", )
    RETURN_NAMES =("caption", ) 
    FUNCTION = "encode"
    CATEGORY = "Florence2"

    def encode(self,tip_pipe,florence2_model,format,max_new_tokens,dir1,dir2,saveDir,text1,text2,template,direction, match_image_size,first_image_shape=None):
        print("执行完成:"+tip_pipe)
        torch.cuda.empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']
        model.to(device)


        task_prompt = "<MORE_DETAILED_CAPTION>"

        prompt = task_prompt

        # image = image.permute(0, 3, 1, 2)

        # 批量读取
        if not os.path.isdir(dir1):
            raise FileNotFoundError(f"Directory '{dir1}' cannot be found.")
        dir_files = os.listdir(dir1)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{dir1}'.")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(dir1, x) for x in dir_files]

    
         # 创建保存目录
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        index1 = 0 
        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            start = time.time()

            #查找两张图片
            # 获取文件名（不包含路径）
            file_name = os.path.basename(image_path)

             # 构造第二张图片的路径
            second_image_path = os.path.join(dir2, file_name)

            # 检查第二张图片是否存在
            if not os.path.isfile(second_image_path):
                print(f"Second image not found for {file_name}")
                continue

             #检查是否已经存在
            file_name_without_ext, _ = os.path.splitext(file_name)
            img_file_name = f"{file_name_without_ext}.{format}"

            # if format != "png":
            #     if input_image.mode == "RGBA":
            #         input_image = input_image.convert("RGB")
            img_save_path = os.path.join(saveDir, img_file_name)
            if os.path.isfile(img_save_path):
                print(f"存在跳过: {img_file_name}")
                index1 = index1 + 1
                continue

            # 打开图片
            input_image = open_image(image_path)
            input_image = ImageOps.exif_transpose(input_image)
            input_image = input_image.convert("RGB")

            second_image = open_image(second_image_path)
            second_image = ImageOps.exif_transpose(second_image)
            second_image = second_image.convert("RGB")
        
            image_pil = input_image
            inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=3,
            )

            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            clean_results = str(results)       
            clean_results = clean_results.replace('</s>', '')
            clean_results = clean_results.replace('<s>', '')

            W, H = image_pil.size
            parsed_answer = processor.post_process_generation(results, task=task_prompt, image_size=(W, H))
            caption = parsed_answer[task_prompt]

             # 提示词
            # file_name_without_ext, _ = os.path.splitext(file_name)
            caption = caption.strip()
            txt_file_name = f"{file_name_without_ext}.txt"
            txt_save_path = os.path.join(saveDir, txt_file_name)
            final_text = template.replace("{caption}", caption).replace("{text1}", text1).replace("{text2}", text2)

            try:
                with open(txt_save_path, 'w', encoding='utf-8') as file:
                    file.write(final_text)
            except IOError as e:
                print(f"保存文件时发生错误: {e}")

             # Check if the batch sizes are different
            image1 = pil2tensor(input_image)
            image2 = pil2tensor(second_image)
            batch_size1 = image1.shape[0]
            batch_size2 = image2.shape[0]

            if batch_size1 != batch_size2:
                # Calculate the number of repetitions needed
                max_batch_size = max(batch_size1, batch_size2)
                repeats1 = max_batch_size // batch_size1
                repeats2 = max_batch_size // batch_size2
                
                # Repeat the images to match the largest batch size
                image1 = image1.repeat(repeats1, 1, 1, 1)
                image2 = image2.repeat(repeats2, 1, 1, 1)

            if match_image_size:
                # Use first_image_shape if provided; otherwise, default to image1's shape
                target_shape = first_image_shape if first_image_shape is not None else image1.shape

                original_height = image2.shape[1]
                original_width = image2.shape[2]
                original_aspect_ratio = original_width / original_height

                if direction in ['left', 'right']:
                    # Match the height and adjust the width to preserve aspect ratio
                    target_height = target_shape[1]  # B, H, W, C format
                    target_width = int(target_height * original_aspect_ratio)
                elif direction in ['up', 'down']:
                    # Match the width and adjust the height to preserve aspect ratio
                    target_width = target_shape[2]  # B, H, W, C format
                    target_height = int(target_width / original_aspect_ratio)
                
                # Adjust image2 to the expected format for common_upscale
                image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
                
                # Resize image2 to match the target size while preserving aspect ratio
                image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
                
                # Adjust image2 back to the original format (B, H, W, C) after resizing
                image2_resized = image2_resized.movedim(1, -1)
            else:
                image2_resized = image2

            # Concatenate based on the specified direction
            if direction == 'right':
                concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
            elif direction == 'down':
                concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
            elif direction == 'left':
                concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
            elif direction == 'up':
                concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height

            input_image = tensor2pil(concatenated_image)

            # 图片
            # img_file_name = f"{file_name_without_ext}.{format}"

            # if format != "png":
            #     if input_image.mode == "RGBA":
            #         input_image = input_image.convert("RGB")
            # img_save_path = os.path.join(saveDir, img_file_name)
            input_image.save(img_save_path)
            end = time.time()
            execution_time = calculate_seconds_difference(start, end)
            temp = f":{execution_time:.3f}s"
            index1 = index1 + 1
            print(str(index1)+"/"+str(len(dir_files)) +":"+temp)

        print("finish结束")    
        model.to(offload_device)
        mm.soft_empty_cache()
        
        return (saveDir,)

class CXH_Ic_lora_Joy_batch:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "JoyPipeline_alpha": ("JoyPipeline_alpha",),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "format": (["png", "jpg"],),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "dir1": ("STRING", {"default": ""}),
                "dir2":  ("STRING", {"default": ""}),
                "saveDir":  ("STRING", {"default": ""}),
                "text1": ("STRING", {"default": "", "multiline": True, "label": "Text Box 1"}),
                "text2": ("STRING", {"default": "", "multiline": True, "label": "Text Box 2"}),
                "template": ("STRING", {"default": "Realistic style, [cloth-on], the image pair highlights a transformation from a clothing sample photo to the effect of actually wearing it. [image1] {caption} [image2] a female model is wearing the cloth from [image1] with {caption}", "multiline": True, "label": ""}),
                "direction": (
                [   'right',
                    'down',
                    'left',
                    'up',
                ],
                {
                "default": 'right'
                }),
                "match_image_size": ("BOOLEAN", {"default": True}),
                }
        }

    RETURN_TYPES = ()
    FUNCTION = "gen"
    OUTPUT_NODE = True
    CATEGORY = "CXH/Images"

    def gen(self,JoyPipeline_alpha,prompt,format,max_new_tokens,dir1,dir2,saveDir,text1,text2,template,direction, match_image_size, first_image_shape=None):

        torch.cuda.empty_cache()

        joy_pipeline =  JoyPipeline_alpha 
        if joy_pipeline.clip_processor == None :
            joy_pipeline.parent.loadCheckPoint()    

        clip_processor = joy_pipeline.clip_processor
        tokenizer = joy_pipeline.tokenizer
        clip_model = joy_pipeline.clip_model
        image_adapter = joy_pipeline.image_adapter
        text_model = joy_pipeline.text_model

        convo = [
                {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
        prompt_tokens = prompt_tokens.squeeze(0)

        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
                0].tolist()
        assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt


        # text_model = joy_two_pipeline.llm.load_llm_model(joy_two_pipeline.model)
        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to('cuda'))



        # 批量读取
        if not os.path.isdir(dir1):
            raise FileNotFoundError(f"Directory '{dir1}' cannot be found.")
        dir_files = os.listdir(dir1)

        # if not os.path.isdir(dir2):
        #     raise FileNotFoundError(f"Directory '{dir2}' cannot be found.")
        # dir_files_2 = os.listdir(dir2)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{dir1}'.")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(dir1, x) for x in dir_files]

    
         # 创建保存目录
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        index1 = 0
        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            start = time.time()

            #查找两张图片
            # 获取文件名（不包含路径）
            file_name = os.path.basename(image_path)

             # 构造第二张图片的路径
            second_image_path = os.path.join(dir2, file_name)

            # 检查第二张图片是否存在
            if not os.path.isfile(second_image_path):
                print(f"Second image not found for {file_name}")
                index1 = index1 + 1
                continue

            #检查是否已经存在
            file_name_without_ext, _ = os.path.splitext(file_name)
            img_file_name = f"{file_name_without_ext}.{format}"

            # if format != "png":
            #     if input_image.mode == "RGBA":
            #         input_image = input_image.convert("RGB")
            img_save_path = os.path.join(saveDir, img_file_name)
            if os.path.isfile(img_save_path):
                print(f"存在跳过 {img_file_name}")
                index1 = index1 + 1
                continue

            # 打开图片
            input_image = open_image(image_path)
            input_image = ImageOps.exif_transpose(input_image)
            input_image = input_image.convert("RGB")

            second_image = open_image(second_image_path)
            second_image = ImageOps.exif_transpose(second_image)
            second_image = second_image.convert("RGB")


            image = input_image.resize((384, 384), Image.LANCZOS)
            pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to('cuda')


            with torch.amp.autocast_mode.autocast('cuda', enabled=True):
                vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
                embedded_images = image_adapter(vision_outputs.hidden_states)
                embedded_images = embedded_images.to('cuda')

            input_embeds = torch.cat([
                convo_embeds[:, :preamble_len],  # Part before the prompt
                embedded_images.to(dtype=convo_embeds.dtype),  # Image
                convo_embeds[:, preamble_len:],  # The prompt and anything after it
            ], dim=1).to('cuda')

            input_ids = torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to('cuda')
            attention_mask = torch.ones_like(input_ids)

            generate_ids = text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
                                            max_new_tokens=max_new_tokens, do_sample=True,
                                            suppress_tokens=None)  # Uses the default which is temp=0.6, top_p=0.9


            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids(
                    "<|eot_id|>"):
                generate_ids = generate_ids[:, :-1]

            caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

            # 提示词
           
            caption = caption.strip()
            txt_file_name = f"{file_name_without_ext}.txt"
            txt_save_path = os.path.join(saveDir, txt_file_name)
            final_text = template.replace("{caption}", caption).replace("{text1}", text1).replace("{text2}", text2)

            try:
                with open(txt_save_path, 'w', encoding='utf-8') as file:
                    file.write(final_text)
            except IOError as e:
                print(f"保存文件时发生错误: {e}")

          
            # Check if the batch sizes are different
            image1 = pil2tensor(input_image)
            image2 = pil2tensor(second_image)
            batch_size1 = image1.shape[0]
            batch_size2 = image2.shape[0]

            if batch_size1 != batch_size2:
                # Calculate the number of repetitions needed
                max_batch_size = max(batch_size1, batch_size2)
                repeats1 = max_batch_size // batch_size1
                repeats2 = max_batch_size // batch_size2
                
                # Repeat the images to match the largest batch size
                image1 = image1.repeat(repeats1, 1, 1, 1)
                image2 = image2.repeat(repeats2, 1, 1, 1)

            if match_image_size:
                # Use first_image_shape if provided; otherwise, default to image1's shape
                target_shape = first_image_shape if first_image_shape is not None else image1.shape

                original_height = image2.shape[1]
                original_width = image2.shape[2]
                original_aspect_ratio = original_width / original_height

                if direction in ['left', 'right']:
                    # Match the height and adjust the width to preserve aspect ratio
                    target_height = target_shape[1]  # B, H, W, C format
                    target_width = int(target_height * original_aspect_ratio)
                elif direction in ['up', 'down']:
                    # Match the width and adjust the height to preserve aspect ratio
                    target_width = target_shape[2]  # B, H, W, C format
                    target_height = int(target_width / original_aspect_ratio)
                
                # Adjust image2 to the expected format for common_upscale
                image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
                
                # Resize image2 to match the target size while preserving aspect ratio
                image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
                
                # Adjust image2 back to the original format (B, H, W, C) after resizing
                image2_resized = image2_resized.movedim(1, -1)
            else:
                image2_resized = image2

            # Concatenate based on the specified direction
            if direction == 'right':
                concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
            elif direction == 'down':
                concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
            elif direction == 'left':
                concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
            elif direction == 'up':
                concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height

            input_image = tensor2pil(concatenated_image)

           
            input_image.save(img_save_path)
            end = time.time()
            execution_time = calculate_seconds_difference(start, end)
            temp = f":{execution_time:.3f}s"
            index1 = index1 + 1
            print(str(index1)+"/"+str(len(dir_files)) +":"+temp)

        print("finish结束")
        joy_pipeline.parent.clearCache()  
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        return (saveDir, )

class CXH_IC_lora_reversal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir1": ("STRING", {"default": ""}),
                "dir2":  ("STRING", {"default": ""}),
                "text_dir":  ("STRING", {"default": ""}),
                "save_dir":  ("STRING", {"default": ""}),
                "slic_start":  ("STRING", {"default": "[image1]"}),
                "slic_end":  ("STRING", {"default": "[image2]"}),
                "format": (["png", "jpg"],),
                "text1": ("STRING", {"default": "", "multiline": True, "label": "Text Box 1"}),
                "text2": ("STRING", {"default": "", "multiline": True, "label": "Text Box 2"}),
                "template": ("STRING", {"default": "Realistic style, [cloth-on], the image pair highlights a transformation from a female model wearing the cloth to its clothing sample photo. [image1] a female model is wearing a cloth with {caption} [image2] the clothing sample photo of what the model is wearing in [image1] with {caption}", "multiline": True, "label": ""}),
                "direction": (
                [   'right',
                    'down',
                    'left',
                    'up',
                ],
                {
                "default": 'right'
                }),
                "match_image_size": ("BOOLEAN", {"default": True}),
                }
            }
    
    RETURN_TYPES = ( "STRING", )
    RETURN_NAMES =("caption", ) 
    FUNCTION = "encode"
    CATEGORY = "Florence2"

    def encode(self,dir1,dir2,text_dir,save_dir,slic_start,slic_end,format,text1,text2,template,direction, match_image_size,first_image_shape=None):

        # 批量读取
        if not os.path.isdir(dir1):
            raise FileNotFoundError(f"Directory '{dir1}' cannot be found.")
        dir_files = os.listdir(dir1)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{dir1}'.")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(dir1, x) for x in dir_files]

    
         # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        index1 = 0 
        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            start = time.time()

            #查找两张图片
            # 获取文件名（不包含路径）
            file_name = os.path.basename(image_path)

            # 构造第二张图片的路径
            second_image_path = os.path.join(dir2, file_name)

            #检查是否已经存在
            file_name_without_ext, _ = os.path.splitext(file_name)
            # 第三个txt
            text_file =  os.path.join(text_dir, file_name_without_ext+".txt")

            # 检查第二张图片是否存在
            if not os.path.isfile(second_image_path) or not os.path.isfile(text_file) :
                print(f"Second image not found for {file_name}")
                continue

            img_file_name = f"{file_name_without_ext}.{format}"

           

            # if format != "png":
            #     if input_image.mode == "RGBA":
            #         input_image = input_image.convert("RGB")
            img_save_path = os.path.join(save_dir, img_file_name)
            if os.path.isfile(img_save_path):
                print(f"存在跳过: {img_file_name}")
                index1 = index1 + 1
                continue

            # 打开图片
            input_image = open_image(image_path)
            input_image = ImageOps.exif_transpose(input_image)
            input_image = input_image.convert("RGB")

            second_image = open_image(second_image_path)
            second_image = ImageOps.exif_transpose(second_image)
            second_image = second_image.convert("RGB")
            
            #打开prompt
            # 使用 open 函数打开文件，模式为 'r' 表示读取模式
            with open(text_file, 'r', encoding='utf-8') as file:
                # 读取文件的所有内容，并存储在变量中
                content = file.read()
            # 找到字符的索引
            start_index = content.index(slic_start) + len(slic_start)
            end_index = content.index(slic_end)  # 不需要加1，因为我们不包含end_char
            
            # 使用切片语法截取字符串
            sliced_string = content[start_index:end_index]

            final_text = template.replace("{caption}", sliced_string).replace("{text1}", text1).replace("{text2}", text2)

            txt_file_name = f"{file_name_without_ext}.txt"
            txt_save_path = os.path.join(save_dir, txt_file_name)
            try:
                with open(txt_save_path, 'w', encoding='utf-8') as file:
                    file.write(final_text)
            except IOError as e:
                print(f"保存文件时发生错误: {e}")

          
            # Check if the batch sizes are different
            image1 = pil2tensor(input_image)
            image2 = pil2tensor(second_image)
            batch_size1 = image1.shape[0]
            batch_size2 = image2.shape[0]

            if batch_size1 != batch_size2:
                # Calculate the number of repetitions needed
                max_batch_size = max(batch_size1, batch_size2)
                repeats1 = max_batch_size // batch_size1
                repeats2 = max_batch_size // batch_size2
                
                # Repeat the images to match the largest batch size
                image1 = image1.repeat(repeats1, 1, 1, 1)
                image2 = image2.repeat(repeats2, 1, 1, 1)

            if match_image_size:
                # Use first_image_shape if provided; otherwise, default to image1's shape
                target_shape = first_image_shape if first_image_shape is not None else image1.shape

                original_height = image2.shape[1]
                original_width = image2.shape[2]
                original_aspect_ratio = original_width / original_height

                if direction in ['left', 'right']:
                    # Match the height and adjust the width to preserve aspect ratio
                    target_height = target_shape[1]  # B, H, W, C format
                    target_width = int(target_height * original_aspect_ratio)
                elif direction in ['up', 'down']:
                    # Match the width and adjust the height to preserve aspect ratio
                    target_width = target_shape[2]  # B, H, W, C format
                    target_height = int(target_width / original_aspect_ratio)
                
                # Adjust image2 to the expected format for common_upscale
                image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
                
                # Resize image2 to match the target size while preserving aspect ratio
                image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
                
                # Adjust image2 back to the original format (B, H, W, C) after resizing
                image2_resized = image2_resized.movedim(1, -1)
            else:
                image2_resized = image2

            # Concatenate based on the specified direction
            if direction == 'right':
                concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
            elif direction == 'down':
                concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
            elif direction == 'left':
                concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
            elif direction == 'up':
                concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height

            input_image = tensor2pil(concatenated_image)

           
            input_image.save(img_save_path)
            end = time.time()
            execution_time = calculate_seconds_difference(start, end)
            temp = f":{execution_time:.3f}s"
            index1 = index1 + 1
            print(str(index1)+"/"+str(len(dir_files)) +":"+temp)

        print("finish结束")
        return (save_dir, )