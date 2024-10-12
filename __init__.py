
from .Joy_caption_node import Joy_caption_load
from .Joy_caption_node import Joy_caption
from .miniCPMv2_6_prompt_generator import CXH_HG_Model_Load,CXH_Min2_6_prompt_Run
from .florence_nodes import CXH_DownloadAndLoadFlorence2Model,CXH_Florence2Run
from .miniCpMV3_4_chat import  CXH_MinCP3_4B_Load,CXH_MinCP3_4B_Chat
from .Joy_caption_alpha import Joy_caption_alpha_load,Joy_caption_alpha_run,Joy_caption_alpha_prompt

NODE_CLASS_MAPPINGS = {
    "Joy_caption_load":Joy_caption_load,
    "Joy_caption":Joy_caption,
    "CXH_HG_Model_Load":CXH_HG_Model_Load,
    "CXH_Min2_6_prompt_Run":CXH_Min2_6_prompt_Run,
    "CXH_DownloadAndLoadFlorence2Model":CXH_DownloadAndLoadFlorence2Model,
    "CXH_Florence2Run":CXH_Florence2Run,
    "CXH_MinCP3_4B_Load":CXH_MinCP3_4B_Load,
    "CXH_MinCP3_4B_Chat":CXH_MinCP3_4B_Chat,
    "Joy_caption_alpha_load":Joy_caption_alpha_load,
    "Joy_caption_alpha_run":Joy_caption_alpha_run,
    "Joy_caption_alpha_prompt":Joy_caption_alpha_prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Joy_caption_load":"Joy_caption_load",
    "Joy_caption":"Joy_caption",
    "CXH_HG_Model_Load":"CXH_HG_Model_Load",
    "CXH_Min2_6_prompt_Run":"CXH_Min2_6_prompt_Run",
    "CXH_DownloadAndLoadFlorence2Model":"CXH_DownloadAndLoadFlorence2Model",
    "CXH_Florence2Run":"CXH_Florence2Run",
    "CXH_MinCP3_4B_Load":"CXH_MinCP3_4B_Load",
    "CXH_MinCP3_4B_Chat":"CXH_MinCP3_4B_Chat",
    "Joy_caption_alpha_load":"Joy_caption_alpha_load",
    "Joy_caption_alpha_run":"Joy_caption_alpha_run",
    "Joy_caption_alpha_prompt":"Joy_caption_alpha_prompt"
}
