import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ex2_open_router.mas_stage1 import Stage_1
from ex2_open_router.mas_stage2 import Stage_2
from termcolor import cprint

model_list = [
    'mistralai/mistral-small-3.2-24b-instruct',
    'mistralai/mistral-small-3.2-24b-instruct:free',
    'moonshotai/kimi-vl-a3b-thinking:free',
    'qwen/qwen2.5-vl-32b-instruct:free',
    'qwen/qwen2.5-vl-72b-instruct:free',
    'google/gemma-3-12b-it:free',
    'google/gemma-3-27b-it:free',
    'meta-llama/llama-3.2-11b-vision-instruct:free'
]

vlm_model = model_list[0]

# --- stage 1 ---
stage1 = Stage_1(
    vlm_model          = vlm_model,
    provider_id        = "DeepInfra",
    exp_image_list_len = 10,
    exp_len            = 2000,
    trail_num          = 't-6',
    # resume_idx         = 278
)
# stage1.predict()

# --- stage 2 ---
stage2 = Stage_2(
    vlm_model    = vlm_model,
    max_tokens   = 1024,
    s1_data_file = stage1.output_file,
)
# stage2.predict()
stage2.evaluate()