
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM

# from vllm import LLM, SamplingParams
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import folder_paths

from .lib.ximg import *
from .lib.xmodel import *

device = "cuda"

class CXH_MinCP3_4B_Pipe:

    def __init__(self):
        self.model = None
        self.tokenizer  =None


class CXH_MinCP3_4B_Load:

    def __init__(self):
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["openbmb/MiniCPM3-4B","openbmb/MiniCPM3-4B-GPTQ-Int4"],), 
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("CXH_MinCP3_4B_Pipe",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "gen"

    def gen(self,model):
        
        self.pipe = CXH_MinCP3_4B_Pipe()

        MODEL_PATH = download_hg_model(model,"LLM")

        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

        self.pipe.model = model
        self.pipe.tokenizer = tokenizer

        return (self.pipe,)

class CXH_MinCP3_4B_Chat:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_MinCP3_4B_Pipe",),
                "prompt":   ("STRING", {"multiline": True, "default": "Provide a detailed description of the details and content contained in the image, and generate a short prompt that can be used for image generation tasks in Stable Diffusion,remind you only need respons prompt itself and no other information."},),
                "top_p":("FLOAT", {"default": 0.7, "min": 0.0, "max": 1, "step": 1}),
                "max_tokens":("INT", {"default": 1024, "min": 10, "max": 4048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,pipe,prompt,top_p,max_tokens,temperature,seed): 

        messages = [
            {"role": "user", "content": prompt},
        ]

        model_inputs = pipe.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

        model_outputs = pipe.model.generate(
            model_inputs,
            max_new_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature
        )

        output_token_ids = [
            model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
        ]

        responses = pipe.tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
        # print(responses)
        return (responses,)
