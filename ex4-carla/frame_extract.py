import json
from random import choice
import os, sys

import cv2
from termcolor import cprint

video_dir = '/data/hiho/raw-carla/jl-2025_08_26'

dest_root_dir = '/data/hiho/raw-carla/extracted-jl-2025_08_26'
if not os.path.exists(dest_root_dir):
    os.makedirs(dest_root_dir)

# --- filter video paths end with agent1 or agent2 only ---
video_path_list = [os.path.join(video_dir, f) for f in sorted(os.listdir(video_dir)) if f.endswith('.mp4') and not f.endswith('sbs.mp4')]


def extract_frames():
    for video_path in video_path_list:
        dest_dir_name = video_path.split('.')[-2].replace('pth_', '')
        # dest_dir_name = dest_dir_name[0:15] + dest_dir_name[-6:]
        # dest_dir_name = dest_dir_name[:-8]
        # print(dest_dir_name)
        dest_dir_url = os.path.join(dest_root_dir, dest_dir_name)
        if not os.path.exists(dest_dir_url):
            os.makedirs(dest_dir_url)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = os.path.join(dest_dir_url, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(out_path, frame)
            frame_idx += 1
        cap.release()
        cprint(f'Extracted frames to: {dest_dir_url}')

# --- generate annotation file ---
def gen_annotation_file():
    round_list = [folder_name.split('.')[-2].replace('pth_', '')[:-8] for folder_name in video_path_list]
    round_list = sorted((set(round_list)))

    anno_data = {}
    round_id  = 0
    for round_name in round_list:
        left_label = f'agent{choice([1, 2])}'
        right_label = 'agent1' if left_label == 'agent2' else 'agent2'
        gt_label = 0 if left_label == 'agent2' else 1
        anno_data[f'r_{round_id}'] = {
            'round_name'    : round_name,
            'left'          : left_label,
            'right'         : right_label,
            'human_label'   : gt_label
        }
        round_id += 1

    with open('ex4-carla/annotations.json', 'w') as f:
        json.dump(anno_data, f, indent=4)
        f.close()


# extract_frames()
gen_annotation_file()
