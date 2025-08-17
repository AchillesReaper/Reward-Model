'''
This script is the first stage of the pipeline -> generating descriptions
expected output: a json file with the descriptions of the robot walking sequences
{
    "meta_data": {
        "task": "walker2d-annotation",
        "image_list_length": 20,
        "total_epochs": 2,
        "s1_prompt": "This sequence of images shows a abstract robot walking. Describe in detail how the robot is walking in terms of balance and how natural it looks."
    },
    "results":{
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


def chat(messages, temperature=0.7, model="gpt-3.5-turbo"):
    try:
        gpt_client = OpenAI(api_key=apiKey_1)
        response = gpt_client.chat.completions.create(
            model       = gpt_model,
            messages    = messages,
            temperature = temperature
        )
        prompt_tokens     = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        return response.choices[0].message.content.strip(), prompt_tokens, completion_tokens
    except Exception as e:
        cprint(f"Error occurred: {e}", 'red')
        return f"API error: {str(e)}", 0, 0


def planner_agent(task):
    messages = [
        {"role": "system", "content": "You are a Planner agent. Break down complex tasks into steps."},
        {"role": "user", "content": [
            {"type": "text", "text": f"The task is: {task}"},
        ]}
    ]
    return chat(messages)


def researcher_agent(step, image_parts):
    messages = [
        {"role": "system", "content": "You are a Researcher agent. Provide factual, detailed answers to specific questions or sub-tasks."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Please research and elaborate on: {step}"},
            *image_parts
        ]},

    ]
    return chat(messages, temperature=gpt_temperature, model=gpt_model)


def critic_agent(task, plan, research_outputs):
    messages = [
        {"role": "system", "content": "You are a Critic agent. Evaluate the reasoning and outputs of multiple agents. Provide constructive feedback and a refined answer."},
        {"role": "user", "content": f"""
            The original task was: {task}
            The plan was:
            {plan}
            The research outputs were:
            {research_outputs}

            Please assess whether the reasoning is sound, and give a refined and well-structured final answer.
            """
        }
    ]
    return chat(messages, temperature=gpt_temperature, model=gpt_model)


def single_sequence_describe(img_folder_path, plan):
    # cprint(exp_question_prompt, "blue")
    # cprint(plan, "blue")
    # sys.exit()
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
    total_prompt_tokens     = 0
    total_completion_tokens = 0
    research_outputs        = []
    for i, step in enumerate(plan[1:-2]):
        step_content, prompt_tokens, completion_tokens = researcher_agent(step, image_parts)
        research_outputs.append(step_content)
        total_prompt_tokens     += prompt_tokens
        total_completion_tokens += completion_tokens

    final_output, prompt_tokens, completion_tokens = critic_agent(exp_question_prompt, plan, research_outputs)
    total_prompt_tokens     += prompt_tokens
    total_completion_tokens += completion_tokens

    return research_outputs, final_output, total_prompt_tokens, total_completion_tokens


def pair_sequence_describe(round_id, round_data, plan):
    epoch_left_folder = f'{trail_folder}/extract/l_{round_data["left"]}'
    epoch_right_folder = f'{trail_folder}/extract/r_{round_data["right"]}'

    # Generate description for left sequence
    left_research_outputs, left_description, left_prompt_tokens, left_completion_tokens = single_sequence_describe(epoch_left_folder, plan)
    # Generate description for right sequence
    right_research_outputs, right_description, right_prompt_tokens, right_completion_tokens = single_sequence_describe(epoch_right_folder, plan)

    return {
        "round_id": round_id,
        "round_data": {
            "left_research_outputs": left_research_outputs,
            "right_research_outputs": right_research_outputs,
            "left_description": left_description,
            "right_description": right_description,
            'tk_usage_s1': {
                "token_usage_left": {
                    "prompt_tokens": left_prompt_tokens,
                    "completion_tokens": left_completion_tokens,
                },
                "token_usage_right": {
                    "prompt_tokens": right_prompt_tokens,
                    "completion_tokens": right_completion_tokens,
                }
            }
        }
    }


def predict():
    # --- meta data ---
    plan, prompt_tokens, completion_tokens = planner_agent(exp_question_prompt)
    plan = [step.strip("- ").strip() for step in plan.split("\n\n") if step.strip()]

    meta_data = {
        'task'              : exp_task,
        'image_list_length' : exp_img_list_len,
        'total_epochs'      : exp_len,
        's1_prompt'         : exp_question_prompt.strip().replace("\n", " "),
        's1_plan'           : plan,
    }
    
    # --- get the annotation data ---
    with open(f'{trail_folder}/annotation.json', 'rb') as f:
        annotation_data = json.load(f)
        f.close()

    # --- Process each image sequence pair and generate description
    results = {}
    parallel_results = Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(pair_sequence_describe)(round_id, round_data, plan)
        for round_id, round_data in tqdm(islice(annotation_data.items(), exp_len), desc="Processing sequences", total=exp_len)
    )

    for item in parallel_results:
        results[item['round_id']] = item['round_data']

    # --- Save to output file ---
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
    full_prompt = "Given sequence of images shows a abstract robot walking, describe in detail how the robot is walking in terms of balance and how natural it looks."

    return full_prompt


if __name__ == "__main__":
    exp_task            = 'walker2d-annotation'     # task to generate question prompt
    exp_img_list_len    = 20                        # number of images in each video
    exp_len             = 1                         # number of trails to process

    gpt_temperature     = 0.7
    gpt_model           = 'gpt-4o'
    # gpt_model           = 'gpt-3.5-turbo'             # model to use for the chat

    trail_folder        = '/home/hiho/Data/uni_rlhf_annotation/walker2d-medium-expert-v2_human_labels'
    output_file         = './ex2_gpt4oMAS-walker2d-UniRLHF/output/walker-2d-annotation_s1.json'

    exp_question_prompt = generate_question_prompt(task=exp_task)

    total_prompt_tokens     = 0
    total_completion_tokens = 0

    predict()