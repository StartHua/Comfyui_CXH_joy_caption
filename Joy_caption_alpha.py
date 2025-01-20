
from huggingface_hub import InferenceClient
from torch import nn
from transformers import  AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import folder_paths
import torchvision.transforms.functional as TVF

from .lib.ximg import *
from .lib.xmodel import *
import re
import time
from datetime import datetime, timedelta

from comfy.model_management import  unload_all_models, soft_empty_cache,get_torch_device

DEVICE = get_torch_device()

def modify_json_value(file_path, key_to_modify, new_value):
  """
  读取 JSON 文件，修改指定 key 的 value 值，并保存修改后的文件。

  Args:
    file_path: JSON 文件路径。
    key_to_modify: 需要修改的 key。
    new_value:  新的 value 值。
  """
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)

    # 查找并修改 key 的 value
    if key_to_modify in data:
      data[key_to_modify] = new_value
    else:
      print(f"Warning: Key '{key_to_modify}' not found in JSON file.")

    # 保存修改后的 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
      json.dump(data, f, indent=4)  # 使用 indent 参数格式化输出

    print(f"Successfully modified '{key_to_modify}' value in '{file_path}'.")

  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{file_path}'.")

CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a descriptive caption for this image in a formal tone.",
		"Write a descriptive caption for this image in a formal tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a formal tone.",
	],
	"Descriptive (Informal)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Training Prompt": [
		"Write a stable diffusion prompt for this image.",
		"Write a stable diffusion prompt for this image within {word_count} words.",
		"Write a {length} stable diffusion prompt for this image.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Booru tag list": [
		"Write a list of Booru tags for this image.",
		"Write a list of Booru tags for this image within {word_count} words.",
		"Write a {length} list of Booru tags for this image.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}

extra_options_parts = [
        "[如果图像中有人物/角色，你必须用name来指代他们。] If there is a person/character in the image you must refer to them as *name* .",
        "[不要包含无法改变的人物/角色信息如:种族、性别等，但要包含可以改变的属性如:发型。] Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
		"[包含关于光线的信息] Include information about lighting.",
		"[包含关于相机角度的信息] Include information about camera angle.",
		"[包含关于是否有水印的信息] Include information about whether there is a watermark or not.",
		"[包含关于是否有JPEG压缩痕迹的信息] Include information about whether there are JPEG artifacts or not.",
		"[如果是照片，你必须包含可能使用的相机类型以及诸如光圈、快门速度、ISO等细节信息] If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
		"[不要包含任何性相关的内容；保持内容适合全年龄] Do NOT include anything sexual; keep it PG.",
		"[不要提及图像的分辨率] Do NOT mention the image's resolution.",
		"[你必须包含关于图像主观审美质量的评价，从低到非常高] You MUST include information about the subjective aesthetic quality of the image from low to very high.",
		"[包含关于图像构图风格的信息，如引导线、三分法或对称性] Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
		"[不要提及图像中的任何文字] Do NOT mention any text that is in the image.",
		"[指明景深以及背景是否聚焦或模糊] Specify the depth of field and whether the background is in focus or blurred.",
		"[如果适用，提及可能使用的人工或自然光源] If applicable, mention the likely use of artificial or natural lighting sources.",
		"[不要使用任何模棱两可的语言] Do NOT use any ambiguous language.",
		"[包含图像是否适合工作场合（sfw）、暗示性的还是不适合工作场合（nsfw）] Include whether the image is sfw, suggestive, or nsfw.",
		"[只描述图像中最重要的元素] ONLY describe the most important elements of the image."
	]

class JoyPipeline_alpha:
    def __init__(self):
        self.clip_model = None
        self.clip_processor =None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None
        self.parent = None
    
    def clearCache(self):
        self.clip_model = None
        self.clip_processor =None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None 
        


class ImageAdapter_alpha(nn.Module):
	def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
		super().__init__()
		self.deep_extract = deep_extract

		if self.deep_extract:
			input_features = input_features * 5

		self.linear1 = nn.Linear(input_features, output_features)
		self.activation = nn.GELU()
		self.linear2 = nn.Linear(output_features, output_features)
		self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
		self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

		# Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
		self.other_tokens = nn.Embedding(3, output_features)
		self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

	def forward(self, vision_outputs: torch.Tensor):
		if self.deep_extract:
			x = torch.concat((
				vision_outputs[-2],
				vision_outputs[3],
				vision_outputs[7],
				vision_outputs[13],
				vision_outputs[20],
			), dim=-1)
			assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
			assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
		else:
			x = vision_outputs[-2]

		x = self.ln1(x)

		if self.pos_emb is not None:
			assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
			x = x + self.pos_emb

		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)

		# <|image_start|>, IMAGE, <|image_end|>
		other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
		assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
		x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

		return x

	def get_eot_embedding(self):
		return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)



class Joy_caption_alpha_load:

    def __init__(self):
        self.model = None
        self.pipeline = JoyPipeline_alpha()
        self.pipeline.parent = self
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2","unsloth/Meta-Llama-3.1-8B-bnb-4bit"],), 
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("JoyPipeline_alpha",)
    FUNCTION = "gen"

    def loadCheckPoint(self):
        # 清除一波
        if self.pipeline != None:
            self.pipeline.clearCache() 

        # Image Adapter
        adapter_path =  os.path.join(folder_paths.models_dir,"Joy_caption_alpha","image_adapter.pt")

        clip_model_path = os.path.join(folder_paths.models_dir,"Joy_caption_alpha","clip_model.pt")

        CHECKPOINT_PATH = os.path.join(folder_paths.models_dir,"Joy_caption_alpha","text_model")
       
         # clip
        model_id = "google/siglip-so400m-patch14-384"
        CLIP_PATH = download_hg_model(model_id,"clip")

        clip_processor = AutoProcessor.from_pretrained(CLIP_PATH) 
        clip_model = AutoModel.from_pretrained(
                CLIP_PATH,
                trust_remote_code=True
            )
        clip_model = clip_model.vision_model

        print("Loading VLM's custom vision model")
        checkpoint = torch.load(clip_model_path, map_location='cpu')
        checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
        clip_model.load_state_dict(checkpoint)
        del checkpoint

        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to("cuda")

        # Tokenizer
        text_model_path = CHECKPOINT_PATH
        LLM_PATH = download_hg_model(self.model, "LLM")
        modify_json_value(os.path.join(text_model_path, "adapter_config.json"), "base_model_name_or_path",
                              LLM_PATH)
        
        print("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, use_fast=False)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        # LLM
        print("Loading LLM")
        print("Loading VLM's custom text model")
        
        # text_model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH , device_map=0, trust_remote_code=True)
        text_model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH, device_map="auto",
                                                              torch_dtype=torch.bfloat16).eval() 
        
        image_adapter = ImageAdapter_alpha(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False) # ImageAdapter(clip_model.config.hidden_size, 4096) 
        image_adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        adjusted_adapter =  image_adapter 
        adjusted_adapter.eval()
        adjusted_adapter.to("cuda")

        self.pipeline.clip_model = clip_model
        self.pipeline.clip_processor = clip_processor
        self.pipeline.tokenizer = tokenizer
        self.pipeline.text_model = text_model
        self.pipeline.image_adapter = adjusted_adapter
    
    def clearCache(self):
         if self.pipeline != None:
              self.pipeline.clearCache()

    def gen(self,model):
        if self.model == None or self.model != model or self.pipeline == None:
            self.model = model
            self.loadCheckPoint()
        return (self.pipeline,)

def remove_brackets_content(text):
    # 使用正则表达式找到所有被 [] 括起来的内容，并将其删除
    result = re.sub(r'\[.*?\]', '', text)
    return result

class Joy_caption_alpha_prompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        options = list(extra_options_parts)
        required = {
                "caption_type": (["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],),
                "caption_length":(["any", "very short", "short", "medium-length", "long", "very long","20","50","80","100","120","250","500"],),
                "name":("STRING", {"multiline": False, "default": ""},),
            }
        for option in options:
            required[option] = ("BOOLEAN", {"default": False})
        return {
            "required": required
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    # def gen(self,caption_type,caption_length,extra_options): 
    def gen(self,**kwargs):
        options_selected = list(kwargs.values())

        caption_type = kwargs["caption_type"]
        caption_length = kwargs["caption_length"]
        name = kwargs["name"]

         
        # 额外选项从第三个参数开始
        extra_options = options_selected[3:]

        length = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")
        
        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

        prompt_str = prompt_str.format(length=caption_length, word_count=caption_length)
        options = list(extra_options_parts)
        for selected, option in zip(extra_options, options):
            if selected:
                prompt_str = prompt_str + remove_brackets_content(option)
        prompt_str = prompt_str.replace("*name*", name)
        print(prompt_str)
        return (prompt_str,)
    
class Joy_caption_alpha_run:

    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "JoyPipeline_alpha": ("JoyPipeline_alpha",),
                "image": ("IMAGE",),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache": ("BOOLEAN", {"default": False}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,JoyPipeline_alpha,image,prompt,max_new_tokens,temperature,cache,low_vram,seed): 

        torch.cuda.empty_cache()

        if low_vram :
            unload_all_models()

        joy_pipeline =  JoyPipeline_alpha 
        if joy_pipeline.clip_processor == None :
            joy_pipeline.parent.loadCheckPoint()    

        clip_processor = joy_pipeline.clip_processor
        tokenizer = joy_pipeline.tokenizer
        clip_model = joy_pipeline.clip_model
        image_adapter = joy_pipeline.image_adapter
        text_model = joy_pipeline.text_model

     

        input_image = tensor2pil(image)

        # Preprocess image
        # pImge = clip_processor(images=input_image, return_tensors='pt').pixel_values
        # pImge = pImge.to(DEVICE)

        image = input_image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to('cuda')

        # Tokenize the prompt
        # prompt = tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        # Embed image

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to('cuda')


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

        if cache == False:
            joy_pipeline.parent.clearCache()  
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            if low_vram:
                unload_all_models()
                soft_empty_cache()

        return (caption.strip(), )
    

# ===============批量打标=============
class Joy_caption_alpha_batch:

    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "JoyPipeline_alpha": ("JoyPipeline_alpha",),
                "img_dir": ("STRING", {"multiline": True, "default": ""},),
                "save_dir":   ("STRING", {"multiline": True, "default": ""},),
                "trigger":   ("STRING", {"multiline": False, "default": "trigger"},),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "format": (["png", "jpg"],),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache": ("BOOLEAN", {"default": False}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,JoyPipeline_alpha,img_dir,save_dir,trigger,prompt,format,max_new_tokens,temperature,cache,low_vram,seed): 

        torch.cuda.empty_cache()
        directory = img_dir
        if low_vram :
            unload_all_models()

        joy_pipeline =  JoyPipeline_alpha 
        if joy_pipeline.clip_processor == None :
            joy_pipeline.parent.loadCheckPoint()    

        clip_processor = joy_pipeline.clip_processor
        tokenizer = joy_pipeline.tokenizer
        clip_model = joy_pipeline.clip_model
        image_adapter = joy_pipeline.image_adapter
        text_model = joy_pipeline.text_model

        # 批量读取
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

         # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

        index1 = 0
        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            start = time.time()
           
            input_image = open_image(image_path)
            input_image = ImageOps.exif_transpose(input_image)
            input_image = input_image.convert("RGB")


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
            lenName = str(index1)
            txt_content = trigger + "," + caption.strip()
            txt_file_name = f"{trigger}_{lenName}.txt"
            txt_save_path = os.path.join(save_dir, txt_file_name)
            try:
                with open(txt_save_path, 'w', encoding='utf-8') as file:
                    file.write(txt_content)
            except IOError as e:
                print(f"保存文件时发生错误: {e}")
            # 图片
            img_file_name = f"{trigger}_{lenName}.{format}"
            if format != "png":
                if input_image.mode == "RGBA":
                    input_image = input_image.convert("RGB")
            img_save_path = os.path.join(save_dir, img_file_name)
            input_image.save(img_save_path)
            end = time.time()
            execution_time = calculate_seconds_difference(start, end)
            temp = f":{execution_time:.3f}s"
            index1 = index1 + 1
            print(str(index1)+"/"+str(len(dir_files)) +":"+temp)
        print("finish结束")

        if cache == False:
            joy_pipeline.parent.clearCache()  
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            if low_vram:
                unload_all_models()
                soft_empty_cache()
        lenName = len(os.listdir(save_dir))
        return (str(lenName/2), )
    
# ===============批量打标=============
def get_subdirectories(directory):
    # 检查目录是否存在
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
    
    # 获取目录中的所有文件夹
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return subdirectories

def get_trigger_from_string(s):
    # Split the string by the underscore character
    parts = s.split('_')
    # Check if the length of the parts is at least 2
    if len(parts) >= 2:
        # Return the second part which is the trigger
        return parts[1]
    else:
        # Return None if the format is not as expected
        return None

class Joy_caption_alpha_batch_Dirs:

    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "JoyPipeline_alpha": ("JoyPipeline_alpha",),
                "img_dir": ("STRING", {"multiline": True, "default": ""},),
                "save_dir":   ("STRING", {"multiline": True, "default": ""},),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "format": (["png", "jpg"],),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache": ("BOOLEAN", {"default": False}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,JoyPipeline_alpha,img_dir,save_dir,prompt,format,max_new_tokens,temperature,cache,low_vram,seed): 

        torch.cuda.empty_cache()
        directory = img_dir
        if low_vram :
            unload_all_models()

        joy_pipeline =  JoyPipeline_alpha 
        if joy_pipeline.clip_processor == None :
            joy_pipeline.parent.loadCheckPoint()    

        clip_processor = joy_pipeline.clip_processor
        tokenizer = joy_pipeline.tokenizer
        clip_model = joy_pipeline.clip_model
        image_adapter = joy_pipeline.image_adapter
        text_model = joy_pipeline.text_model

        # 批量读取
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
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
        subdirs = get_subdirectories(directory)

        for subdir in subdirs:
            print("开始文件夹:"+subdir)
            subdir_path = os.path.join(directory, subdir) 
            if not os.path.isdir(subdir_path):
                continue
            dir_files = os.listdir(subdir_path)
            if len(dir_files) == 0:
                raise FileNotFoundError(f"No files in directory '{directory}'.")

            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

            dir_files = sorted(dir_files)
            dir_files = [os.path.join(subdir_path, x) for x in dir_files]

            # 创建保存目录
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(os.path.join(save_dir,subdir)):
                os.makedirs(os.path.join(save_dir,subdir))

            index1 = 0
            for image_path in dir_files:
                if os.path.isdir(image_path) and os.path.ex:
                    continue
                start = time.time()
                # print(image_path)
                input_image = open_image(image_path)
                input_image = ImageOps.exif_transpose(input_image)
                input_image = input_image.convert("RGB")


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
                lenName = str(index1)
                trigger = get_trigger_from_string(subdir)
                if trigger is not None:
                    txt_content = trigger + "," + caption.strip()
                    txt_file_name = f"{trigger}_{lenName}.txt"
                    txt_save_path = os.path.join(save_dir,subdir, txt_file_name)
                    img_file_name = f"{trigger}_{lenName}.{format}"
                else:
                    txt_content = caption.strip()
                    txt_file_name = f"{lenName}.txt"
                    txt_save_path = os.path.join(save_dir,subdir, txt_file_name)
                    img_file_name = f"{lenName}.{format}"
                try:
                    with open(txt_save_path, 'w', encoding='utf-8') as file:
                        file.write(txt_content)
                except IOError as e:
                    print(f"保存文件时发生错误: {e}")
                # 图片
                
                if format != "png":
                    if input_image.mode == "RGBA":
                        input_image = input_image.convert("RGB")
                img_save_path = os.path.join(save_dir,subdir, img_file_name)
                input_image.save(img_save_path)
                end = time.time()
                execution_time = calculate_seconds_difference(start, end)
                temp = f":{execution_time:.3f}s"
                index1 = index1 + 1
                print(str(index1)+"/"+str(len(dir_files)) +":"+temp)
            print("结束"+subdir)
            index1 = 0

        if cache == False:
            joy_pipeline.parent.clearCache()  
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            if low_vram:
                unload_all_models()
                soft_empty_cache()
        lenName = len(os.listdir(save_dir))
        return (str(lenName/2), )
    
