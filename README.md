![1724901350282](https://github.com/user-attachments/assets/c9d9cd10-fbd6-4aeb-91b6-f2740c3998cc)

(1).基于comfyui节点图片放推(Recommended based on comfyui node pictures)

    1.Joy_caption

    2.miniCPMv2_6_prompt_generator

    3.florence2

(2).安装(Installation)：

  1.（Comfyui evn python.exe） python -m pip install -r requirements.txt or click install_req.bat

  注意：transformers 版本不能太低（Note: The version of transformers cannot be too low）

  2. 下载模型或者运行comfyui自动下载模型到合适文件夹(Download the model or run Comfyui to automatically download the model to the appropriate folder)

(3) Joy_caption

    1.运行自动下载模型(推荐手动下载) Run automatic download model (manual download recommended)

      (1).https://huggingface.co/google/siglip-so400m-patch14-384 放到(put in)clip/siglip-so400m-patch14-384
      

        ![1724901434148](https://github.com/user-attachments/assets/12ad9627-e121-4bc8-98cc-313fa491bde4)

    
      (2). https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit 放到(put in)LLM/Meta-Llama-3.1-8B-bnb-4bit
      
      
        ![1724901495135](https://github.com/user-attachments/assets/3cac31a7-8150-4d78-96d1-8aa3198fe572)


      (3).必须手动下载(Must be downloaded manually):https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6   (put in)Joy_caption 
      
        ![1724901527482](https://github.com/user-attachments/assets/e8ec1be6-a96c-4e73-9422-7bcdafb8f1d4)

(4).例子：worflow/Min2.6+joy+Florence2.json



