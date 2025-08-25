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
import sys, os, json, requests
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List

from termcolor import cprint
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

from api_key import apiKey_openRouter_1 as API_KEY_REF

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


class Stage_2():
    def __init__(
        self,
        api_key            = API_KEY_REF,
        vlm_model          = "qwen/qwen2.5-vl-32b-instruct:free",
        model_temperature  = 0.7,
        max_tokens         = 2048,
        exp_task_name      = "walker2d-annotation",
        s1_data_file       = "./ex2_open_router/qwen-qwen2.5-vl-32b-instruct-free/output/mas_s1_walker2d-annotation.json",
        s2_task            = "You are given two description representing how a abstract robot walking. The goal of the walking robot is to move as much to the right as possible while minimizing energy costs. Choose which robot achieves the goal better. Respond 0 for left and 1 for right. Respond -1 if both videos perform similarly."
    ):
        self.api_key           = api_key
        self.vlm_model         = vlm_model
        self.model_temperature = model_temperature
        self.max_tokens        = max_tokens
        self.exp_task_name     = exp_task_name
        self.s1_data_file      = s1_data_file
        self.s2_task           = s2_task

        
        parser = PydanticOutputParser(pydantic_object=ClassificationResponse)
        format_instructions = parser.get_format_instructions()
        self.s2_task += f"\n------\n{format_instructions}"

        # self.output_file       = f'./ex2_open_router/{self.vlm_model.replace("/", "-").replace(":", "-")}/output/mas_s2_{self.exp_task_name}.json'
        self.output_file       = self.s1_data_file.replace('mas_s1_', 'mas_s2_')


    def predict(self):
        with open(self.s1_data_file, 'r') as f:
            s1_data = json.load(f)
        f.close()

        # --- Process each image pair and generate predictions ---
        results = {}
        parallel_results = Parallel(n_jobs=4, backend='multiprocessing')(
            delayed(self.compare_single_pair)(
                round_id, 
                {
                    "left"  : round_data['left_description'],
                    "right" : round_data['right_description']
                }
            ) for round_id, round_data in tqdm(s1_data["results"].items(), desc="Processing descriptions", total=len(s1_data["results"]))
        )

        for item in parallel_results:
            results[item["round_id"]] = {
                **s1_data["results"][item["round_id"]],  # Include original descriptions and token usage from stage 1
                "choice": item["round_data"]["choice"],
                "reason": item["round_data"]["reason"],
                "tk_usage_s2": item["round_data"]["tk_usage_s2"]
            }

        # --- Save predictions to output JSON file ---
        meta_data = {
            **s1_data['meta_data'],  # Include metadata from stage 1
            "stage_2": {
                'task'  : self.s2_task,
                'model' : {
                    'name': self.vlm_model,
                    'temperature': self.model_temperature,
                    'max_tokens': self.max_tokens
                }
            }
        }

        with open(self.output_file, 'w') as f:
            json.dump({
                "meta_data": meta_data,
                "results": results
            }, f, indent=4)
            cprint(f"Results saved to {self.output_file}", 'green')


    def compare_single_pair(self, round_id, description_pair):
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': self.s2_task},
                    {'type': 'text', 'text': f'LHS robot: {description_pair["left"]}'},
                    {'type': 'text', 'text': f'RHS robot: {description_pair["right"]}'}
                ]
            },
        ]
        response = self.chat(messages)
        try:
            raw_response = response['response']
            json_start = raw_response.index('{')
            json_end = raw_response.index('}') + 1
            json_str = raw_response[json_start:json_end]
            parsed_response = ClassificationResponse.model_validate(json.loads(json_str))
            parsed_response = parsed_response.model_dump()
            result = {
                "choice": parsed_response['choice'],
                "reason": parsed_response['reason'],
                "tk_usage_s2": response['token_usage']
            }
        except Exception as e:
            cprint(f"Error parsing response: {e}", 'red')
            # If parsing fails, return the raw response and token usage
            result = {
                "choice": -1,  # Default to equal
                "reason": "Error occurred during processing",
                "tk_usage_s2": response['token_usage']
            }
        return {
            "round_id": round_id,
            "round_data": result,
        }


    def chat(self, messages):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            'model'       : self.vlm_model,
            'messages'    : messages,
            'max_tokens'  : self.max_tokens,
            'temperature' : self.model_temperature,
        }
        response = requests.post(url, headers=headers, json=payload)
        try:
            response_content = response.json()['choices'][0]['message']['content']
            token_usage      = response.json()['usage']
            return {'response': response_content, 'token_usage': token_usage}
        except Exception as e:
            cprint(messages, 'red')
            cprint(f'response:\n {response.text}', 'blue')
            sys.exit(f"API error: {str(e)}")


    def evaluate(self):
        '''calaulate the accuracy'''
        if self.output_file is None or not os.path.exists(self.output_file):
            sys.exit("Output file not found. Please run prediction first.")
        
        eval_output = []
        with open(self.output_file, 'r') as f:
            s2_data = json.load(f)
            s2_results = s2_data['results']
            for trail in s2_results:
                human_label = s2_results[trail]['human_label']
                model_label = s2_results[trail]['choice']
                score = 1 if human_label == model_label else 0
                eval_output.append({
                    'trail': trail,
                    'human_label': human_label,
                    'model_label': model_label,
                    'score': score
                })
            f.close()
        
        human_label_set = set([item['human_label'] for item in eval_output])
        prediction_summary = {}
        for label in human_label_set:
            prediction_summary[label] = {
                'gt_amount': sum([item['human_label'] == label for item in eval_output]),
                'predicted_amount': sum([item['model_label'] == label for item in eval_output]),
                'correct_amount': sum([item['score'] == 1 for item in eval_output if item['human_label'] == label])
            }

        accuracy = sum([item['score'] for item in eval_output]) / len(eval_output)
        cprint(f"Accuracy: {accuracy:.2f}", 'green')

        df = pd.DataFrame(prediction_summary).T
        df = df.sort_index(ascending=True)
        print(df)

        report = {
            'accuracy': accuracy,
            'prediction_summary': prediction_summary,
            'eval_output': eval_output
        }

        eval_output_file = self.output_file.replace("mas_s2", "mas_s2_eval")
        with open(eval_output_file, 'w') as f:
            json.dump(report, f, indent=4)
            cprint(f"Results saved to {eval_output_file}", 'green')
