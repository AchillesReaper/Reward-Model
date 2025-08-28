import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sa_stage1 import Stage_1

from termcolor import cprint


# ---------- Single Agent ----------
# --- stage 1 offline---
stage1 = Stage_1(
    is_offline         = True,
    model_id           = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    exp_image_list_len = 10,
    exp_len            = 50,
    trail_num          = 't-1',
    exp_task_name      = "carla_early",
    exp_task           = "Given a sequence of images showing a ego car driving. Analyse and describe in detail about the driving behavior.",
    trail_folder       = '/data/hiho/raw-carla/extracted-jl-2025_08_26',
)
stage1.predict()


# --- stage 1 ---
# stage1 = Stage_1(
#     vlm_model          = model_list[1],
#     # provider_id        = "DeepInfra",
#     exp_image_list_len = 8,
#     resume_idx         = 47, 
#     exp_len            = 50,
#     trail_num          = 't-10',
#     exp_task_name      = "carla_early",
#     exp_task           = "Given a sequence of images showing a ego car driving. Analyse and describe in detail about the driving behavior.",
#     trail_folder       = '/data/hiho/raw-carla/extracted-jl-2025_08_26',
# )
# stage1.predict()


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