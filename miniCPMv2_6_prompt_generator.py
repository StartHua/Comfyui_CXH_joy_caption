
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import folder_paths

from .lib.ximg import *
from .lib.xmodel import *

class CXH_Hg_Pipe:

    def __init__(self):
        self.text_model = None
        self.tokenizer  =None


class CXH_HG_Model_Load:

    def __init__(self):
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["pzc163/MiniCPMv2_6-prompt-generator"],), 
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("CXH_Hg_Pipe",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "gen"

    def gen(self,model):
        
        self.pipe = CXH_Hg_Pipe()

        MODEL_PATH = download_hg_model(model,"LLM")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
        text_model.eval()

        self.pipe.text_model = text_model
        self.pipe.tokenizer = tokenizer

        return (self.pipe,)

class CXH_Min2_6_prompt_Run :

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_Hg_Pipe",),
                "image": ("IMAGE",),
                "prompt":   ("STRING", {"multiline": True, "default": "Provide a detailed description of the details and content contained in the image, and generate a short prompt that can be used for image generation tasks in Stable Diffusion,remind you only need respons prompt itself and no other information."},),
                "max_tokens":("INT", {"default": 1024, "min": 10, "max": 4048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,pipe,image,prompt,max_tokens,temperature,seed): 

        image = tensor2pil(image)
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
            

        return (generated_text,)
