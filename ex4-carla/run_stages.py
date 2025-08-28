import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sa_stage1 import Stage_1
from sa_stage2 import Stage_2
from mas_stage1 import Stage_1 as Stage_1_MAS
from termcolor import cprint

model_list = [
    'mistralai/mistral-small-3.2-24b-instruct',
    'mistralai/mistral-small-3.2-24b-instruct:free',
    'moonshotai/kimi-vl-a3b-thinking:free',
    'qwen/qwen2.5-vl-32b-instruct:free',
    # 'qwen/qwen2.5-vl-72b-instruct:free',
    # 'google/gemma-3-12b-it:free',
    # 'google/gemma-3-27b-it:free',
    # 'meta-llama/llama-3.2-11b-vision-instruct:free'
]


# --- stage 1 ---
# mas1 = Stage_1_MAS(
#     vlm_model          = vlm_model,
#     # provider_id        = "DeepInfra",
#     max_tokens         = 2048,
#     exp_image_list_len = 8,
#     # resume_idx         = 4, 
#     exp_len            = 40,
#     trail_num          = 't-6',
#     exp_task_name      = "carla_early",
#     exp_task           = "Given a sequence of images, in chronological order, showing a ego car driving. Analyse and describe in detail about the driving behavior.",
#     trail_folder       = '/data/hiho/raw-carla/extracted-jl-2025_08_26',
# )
# mas1.predict()


# --- stage 2 ---
# mas2 = Stage_2(
#     vlm_model       = stage1.vlm_model,
#     max_tokens      = 1024,
#     s1_data_file    = stage1.output_file,
#     exp_task_name   = stage1.exp_task_name,
#     s2_task         = "You are given two description representing the driving behavior of an ego car. Choose which car drives better. Respond 0 for the left and 1 for the right. Respond -1 if both cars perform similarly."
# )
# mas2.predict()
# mas2.evaluate()




# ---------- Single Agent ----------
# --- stage 1 ---
stage1 = Stage_1(
    vlm_model          = model_list[1],
    # provider_id        = "DeepInfra",
    exp_image_list_len = 8,
    resume_idx         = 47, 
    exp_len            = 50,
    trail_num          = 't-10',
    exp_task_name      = "carla_early",
    exp_task           = "Given a sequence of images showing a ego car driving. Analyse and describe in detail about the driving behavior.",
    trail_folder       = '/data/hiho/raw-carla/extracted-jl-2025_08_26',
)
stage1.predict()


# --- stage 2 ---
# stage2 = Stage_2(
#     vlm_model       = stage1.vlm_model,
#     max_tokens      = 1024,
#     s1_data_file    = stage1.output_file,
#     exp_task_name   = stage1.exp_task_name,
#     s2_task         = "You are given two description representing the driving behavior of an ego car. Choose which car drives better. Respond 0 for the left and 1 for the right. Respond -1 if both cars perform similarly."
# )
# stage2.predict()
# stage2.evaluate()