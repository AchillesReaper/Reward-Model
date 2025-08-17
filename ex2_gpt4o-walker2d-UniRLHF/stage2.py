'''
This script is the Second stage of the pipeline -> make human preference based on descriptions
expected output: a json file with the model predicted preference based on the descriptions
{
    "round_id": {
        "left": number,
        "right": number,
        "human_label": -1|0|1,
        "left_description": "description of the left sequence",
        "right_description": "description of the right sequence",
        "choice": 0|1|-1,  # 0 for left, 1 for right, -1 for equal
        "reason": "reasoning for the choice",
        "tk_usage_s2": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        }
'''
import sys, os, json, base64

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List

from PIL import Image
from termcolor import cprint
from joblib import Parallel, delayed
from tqdm import tqdm
from io import BytesIO

from api_key import apiKey_1, apiKey_3    
from openai import OpenAI

from itertools import islice

client = OpenAI(api_key=apiKey_1)

class ClassificationResponse(BaseModel):
    choice: int = Field(..., description="Classification choice: 0 for left, 1 for right, -1 for equal")
    reason: str = Field(..., min_length=10, description="Detailed reasoning for classification")

    @field_validator('choice')
    def validate_classification(cls, v):
        valid_classes = [0,1,-1]  # 0 for left, 1 for right, -1 for equal
        if v not in valid_classes:
            raise ValueError(f'choice must be one of {valid_classes}')
        return v
    
    @field_validator('reason')
    def validate_reasoning(cls, v):
        if len(v.split()) < 5:
            raise ValueError('Reasoning must contain at least 5 words')
        return v

def compare_single_pair(round_id, description_pair):
    # cprint(f"Processing round {round_id} with descriptions: {description_pair}", 'blue')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": exp_question_prompt},
            {"role": "user", "content": f"LHS robot: {description_pair['left']}"},
            {"role": "user", "content": f"RHS robot: {description_pair['right']}"}
        ],
        max_tokens=512
    )
    tk_usage_s2 = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    try:
        raw_response = response.choices[0].message.content
        json_start = raw_response.index('{')
        json_end = raw_response.index('}') + 1
        json_str = raw_response[json_start:json_end]
        parsed_response = ClassificationResponse.model_validate(json.loads(json_str))
        parsed_response = parsed_response.model_dump()
        result = {
            "choice": parsed_response['choice'],
            "reason": parsed_response['reason'],
            "tk_usage_s2": tk_usage_s2
        }
    except Exception as e:
        cprint(f"Error parsing response: {e}", 'red')
        # If parsing fails, return the raw response and token usage
        result = {
            "choice": -1,  # Default to equal
            "reason": "Error occurred during processing",
            "tk_usage_s2": tk_usage_s2
        }
    return {
        "round_id": round_id,
        "round_data": result,
    }


def predict():
    # Load input JSON file
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    f.close()
    # Process each image pair and generate predictions
    results = {}

    for round_id, round_data in tqdm(input_data["results"].items(), desc="Processing descriptions", total=len(input_data["results"])):
        description_pair = {
            "left": round_data['left_description'],
            "right": round_data['right_description']
        }

        try:
            item = compare_single_pair(round_id, description_pair)
            results[item["round_id"]] = {
                **round_data, 
                "choice": item["round_data"]["choice"],
                "reason": item["round_data"]["reason"],
                "tk_usage_s2": item["round_data"]["tk_usage_s2"]
            }
        except Exception as e:
            cprint(f"Error processing {round_id}: {e}", 'red')
            continue

    meta_data = {
        **input_data['meta_data'],  # Include metadata from stage 1
        "s2_prompt": exp_question_prompt,
    }

    # Save predictions to output JSON file
    with open(output_file, 'w') as f:
        json.dump({
            "meta_data": meta_data,
            "results": results
        }, f, indent=4)
        cprint(f"Results saved to {output_file}", 'green')


def generate_question_prompt():    
    parser = PydanticOutputParser(pydantic_object=ClassificationResponse)
    format_instructions = parser.get_format_instructions()

    # ---------- full prompt ----------
    full_prompt = f"""
    You are given two description representing how a abstract robot walking.
    The goal of the walking robot is to move as much to the right as possible while minimizing energy costs.
    Check if the robot is 
    1) about to fall or 
    2) walking abnormally (e.g., walking on one leg, slipping) or
    3) maintains a normal standing posture for a longer time or 
    4) travels a greater distance.
    ---
    Choose which robot is more helpful for achieving the goal of the agent. respond 0 for left and 1 for right.
    Choose equal if both videos perform similarly. respond -1 for equal.
    {format_instructions}

    """

    return full_prompt

if __name__ == "__main__":
    input_file = 'ex2_gpt4o-walker2d-UniRLHF/output/walker-2d-annotation_s1.json'
    output_file = input_file.replace('s1.json', 's2.json')
    exp_question_prompt = generate_question_prompt()

    if not os.path.exists(output_file): os.makedirs(os.path.dirname(output_file), exist_ok=True)

    predict()