.2024-9-6 florence2 Add Florence-2-large-PromptGen-v1.5

.2024-9-6 florence2 Add Florence-2-base-PromptGen-v1.5 

.2024-9-2 更新批量打标案例(Update batch marking cases) 速度：florence2<min2.6<joy

![1724901350282](https://github.com/user-attachments/assets/c9d9cd10-fbd6-4aeb-91b6-f2740c3998cc)

(1).基于comfyui节点图片放推(Recommended based on comfyui node pictures)

    1.Joy_caption

    2.miniCPMv2_6_prompt_generator

    3.florence2

(2).安装(Installation)：

  1.（Comfyui evn python.exe） python -m pip install -r requirements.txt or click install_req.bat

  注意：transformers 版本不能太低（Note: The version of transformers cannot be too low）

  2. 下载模型或者运行comfyui自动下载模型到合适文件夹(Download the model or run Comfyui to automatically download the model to the appropriate folder)

(3) 模型安装（Install model）

   1).Joy_caption

   .运行自动下载模型(推荐手动下载) Run automatic download model (manual download recommended)
   
    1.https://huggingface.co/google/siglip-so400m-patch14-384 放到(put in)clip/siglip-so400m-patch14-384
      
![1724901434148](https://github.com/user-attachments/assets/12ad9627-e121-4bc8-98cc-313fa491bde4)

    
    2. https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit 放到(put in)LLM/Meta-Llama-3.1-8B-bnb-4bit
      
![1724901495135](https://github.com/user-attachments/assets/3cac31a7-8150-4d78-96d1-8aa3198fe572)


    3.必须手动下载(Must be downloaded manually):https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6   (put in)Joy_caption 
      
![1724901527482](https://github.com/user-attachments/assets/e8ec1be6-a96c-4e73-9422-7bcdafb8f1d4)

 2).MiniCPMv2_6-prompt-generator + CogFlorence
 
 https://huggingface.co/pzc163/MiniCPMv2_6-prompt-generator
 
 https://huggingface.co/thwri/CogFlorence-2.2-Large
 
 ![1724902196890](https://github.com/user-attachments/assets/22373c22-8083-4b3f-af10-774d86560f16)

 Run with:flux1-dev-Q8_0.gguf

 ![e8ad7fa14f807184a99ea23b31e8a60](https://github.com/user-attachments/assets/178ee440-919e-4b28-b1bd-c2c1e2e0ceb4)

 ![1724897220972](https://github.com/user-attachments/assets/ac3c072d-dccc-4f29-bcbd-45c7945407be)

 ![1724897584034](https://github.com/user-attachments/assets/584adc69-3e0d-4cb9-8392-0fe337dc34a2)








