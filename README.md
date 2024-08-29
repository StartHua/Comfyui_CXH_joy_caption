(1).基于comfyui节点图片放推(Recommended based on comfyui node pictures)
    1.Joy_caption

    2.miniCPMv2_6_prompt_generator

    3.florence2

(2).安装(Installation)：

  1.（Comfyui evn python.exe） python -m pip install -r requirements.txt or click install_req.bat

  注意：transformers 版本不能太低（Note: The version of transformers cannot be too low）

  2. 下载模型或者运行comfyui自动下载模型到合适文件夹(Download the model or run Comfyui to automatically download the model to the appropriate folder)

(3) Joy_caption节点模型

    1.运行自动下载模型(推荐手动下载)

      (1).https://huggingface.co/google/siglip-so400m-patch14-384 放到clip/siglip-so400m-patch14-384

    ![1](https://github.com/user-attachments/assets/16fff9e2-2109-44fc-8536-b0fe419bb433)


      (2).推荐下载 https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit（如果你有A100 可以考虑下载meta-llama/Meta-Llama-3.1-8B）放到LLM/Meta-Llama-3.1-8B-bnb-4bit
      
    ![2](https://github.com/user-attachments/assets/ac20590b-2bf2-4768-af2e-860e3d8a31e3)


      (3).必须手动下载:https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6   放到Joy_caption 下

    ![3](https://github.com/user-attachments/assets/237dd9fd-1c37-40cc-a287-086cf5618b6e)


(4).例子：worflow/Min2.6+joy+Florence2.json



