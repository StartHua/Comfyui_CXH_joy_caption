
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModelForVision2Seq,CLIPImageProcessor, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import folder_paths
import time
import re

from .lib.ximg import *
from .lib.xmodel import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CXH_SmolVlm_Pipe:

    def __init__(self):
        self.model = None
        self.processor  =None

class CXH_SmolVlm_Load:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["HuggingFaceTB/SmolVLM-Instruct"],), 
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("CXH_SmolVlm_Pipe",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "gen"
    def gen(self,model): 
        self.pipe = CXH_SmolVlm_Pipe()

        MODEL_PATH = download_hg_model(model,"LLM")
        print(MODEL_PATH)

        # Initialize processor and model
        processor = AutoProcessor.from_pretrained(MODEL_PATH,trust_remote_code=True)
        model1 = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
        ).to(DEVICE)
        

        self.pipe.model = model1
        self.pipe.processor = processor
        return (self.pipe,)

class CXH_SmolVlm_Run :

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_SmolVlm_Pipe",),
                "image": ("IMAGE",),
                "prompt":   ("STRING", {"multiline": True, "default": "Provide a detailed description of the details and content contained in the image, and generate a short prompt that can be used for image generation tasks in Stable Diffusion,remind you only need respons prompt itself and no other information."},),
                "max_tokens":("INT", {"default": 1024, "min": 10, "max": 4048, "step": 1}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,pipe,image,prompt,max_tokens,seed): 

        image = tensor2pil(image)
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        # Prepare inputs
        prompt = pipe.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = pipe.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)

        # Generate outputs
        generated_ids = pipe.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_texts = pipe.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        print(generated_texts[0])
        pattern = re.compile(r"Assistant:\s*(.*)")
        match = pattern.search(generated_texts[0])

        if match:
            number = match.group(1)
            return (number,)
        else:
            print("No number found.")
            return (generated_texts[0],)

