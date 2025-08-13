'''
This script is the first stage of the pipeline -> generating descriptions
expected output: a json file with the descriptions of the robot walking sequences
{
    "round_id": {
        "left": number,
        "right": number,
        "human_label": -1|0|1,
        "left_description": "description of the left sequence",
        "right_description": "description of the right sequence"
        "token_usage_left": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        },
        "token_usage_right": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        }
    }
}
'''

import sys, os, json, base64

from PIL import Image
from termcolor import cprint
from joblib import Parallel, delayed
from tqdm import tqdm
from io import BytesIO

from api_key import apiKey_1
from openai import OpenAI

from itertools import islice

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_reduced_img_sequence(image_folder, num_images=20):
    """Get a reduced sequence of images from the folder."""
    image_paths_list = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    result_path_list = []
    if len(image_paths_list) > num_images:
        step = len(image_paths_list) // num_images
        for i in range(0, len(image_paths_list), step):
            result_path_list.append(image_paths_list[i])
    else:
        result_path_list = image_paths_list
    print(f"Reduced image sequence to {len(result_path_list)} images from {len(image_paths_list)} original images.")
    return result_path_list


def single_sequence_describe(img_folder_path):
    client = OpenAI(api_key=apiKey_1)
    # ---- prepare the image parts ----
    image_path_list = get_reduced_img_sequence(img_folder_path)

    image_parts = []
    # Convert all images to OpenAI vision format
    for img_path in image_path_list:
        base64_img = image_to_base64(img_path)
        image_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}"
            }
        })

    # --- Call GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": exp_question_prompt},
                    *image_parts  # Append all image content here
                ]
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content, response.usage


def pair_sequence_describe(round_id, round_data,):
    epoch_left_folder = f'{trail_folder}/extract/l_{round_data["left"]}'
    epoch_right_folder = f'{trail_folder}/extract/r_{round_data["right"]}'
    
    # Generate description for left sequence
    left_description, left_token_usage = single_sequence_describe(epoch_left_folder)
    # Generate description for right sequence
    right_description, right_token_usage = single_sequence_describe(epoch_right_folder)

    return {
        "round_id": round_id,
        "round_data": {
            "left_description": left_description,
            "right_description": right_description,
            'tk_usage_s1': {
                "token_usage_left": {
                    "prompt_tokens": left_token_usage.prompt_tokens,
                    "completion_tokens": left_token_usage.completion_tokens,
                    "total_tokens": left_token_usage.total_tokens
                },
                "token_usage_right": {
                    "prompt_tokens": right_token_usage.prompt_tokens,
                    "completion_tokens": right_token_usage.completion_tokens,
                    "total_tokens": right_token_usage.total_tokens
                }
            }
        }
    }

def predict():
    # --- get the annotation data ---
    with open(f'{trail_folder}/annotation.json', 'rb') as f:
        annotation_data = json.load(f)
        f.close()

    # --- Process each image sequence pair and generate description
    results = {}
    parallel_results = Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(pair_sequence_describe)(round_id, round_data)
        for round_id, round_data in tqdm(annotation_data.items(), desc="Processing sequences", total=len(annotation_data))
        # for round_id, round_data in tqdm(islice(annotation_data.items(), exp_len), desc="Processing sequences", total=exp_len)
    )

    for item in parallel_results:
        results[item['round_id']] = item['round_data']

    # --- Save results to output file ---
    meta_data = {
        'task': exp_task,
        'image_list_length': exp_img_list_len,
        'total_epochs': len(annotation_data),
        's1_prompt': exp_question_prompt
    }
    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        json.dump({
            'meta_data': meta_data,
            'results': results
        }, f, indent=4)
        f.close()
    cprint(f"Descriptions saved to {output_file}", 'green')


def generate_question_prompt(task):    
    full_prompt = "This sequence of images shows a abstract robot walking. Describe in detail how the robot is walking in terms of balance and how natural it looks."
    # ---------- full prompt ----------

    full_prompt += """
    The goal of the walking robot is to move as much to the right as possible while minimizing energy costs.
    Check if the robot is 
    1) about to fall or 
    2) walking abnormally (e.g., walking on one leg, slipping) or
    3) maintains a normal standing posture for a longer time or 
    4) travels a greater distance.
    """

    return full_prompt


if __name__ == "__main__":
    exp_task = 'walker2d-annotation'       # task to generate question prompt
    exp_img_list_len = 20                  # number of images in each video
    exp_len = 2                         # number of trails to process

    trail_folder = '/home/hiho/Data/uni_rlhf_annotation/walker2d-medium-expert-v2_human_labels'
    output_file = './ex2_gpt4o-walker2d-UniRLHF/output/walker-2d-annotation_s1.json'

    exp_question_prompt = generate_question_prompt(task=exp_task)

    predict()