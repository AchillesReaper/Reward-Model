import sys, os, json, base64, requests, re
from PIL import Image
from termcolor import cprint
from joblib import Parallel, delayed
from itertools import islice
from io import BytesIO
from tqdm import tqdm

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List

from api_key import apiKey_openRouter_1 as API_KEY_REF


# class CriticOutput(BaseModel):
#     critique: str = Field(..., min_length=10, description="Detailed final output produced by the critic agent.")

#     @field_validator('critique')
#     def validate_critique(cls, v):
#         if len(v.split()) < 5:
#             raise ValueError('Critique must contain at least 5 words')
#         return v
class ResponseFormat():
    planner = {
        "type": "json_schema",
        'json_schema': {
            'name'  : 'planner',
            'strict': True,
            'schema': {
                'type'      : 'object',
                'properties': {
                    'task':{
                        'type': 'string',
                        'description': 'The task to be performed'
                    },
                    'steps': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'description': 'A step in the task'
                        }
                    }
                },
                'required': ['task', 'steps']
            }
        }
    }
    researcher = {
        "type": "json_schema",
        'json_schema': {
            'name'  : 'researcher',
            'strict': True,
            'schema': {
                'type'      : 'object',
                'properties': {
                    'findings': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'description': 'A finding in the research'
                        }
                    }
                },
                'required': ['findings']
            }
        }
    }
    critic = {
        "type": "json_schema",
        'json_schema': {
            'name'  : 'critic',
            'strict': True,
            'schema': {
                'type'      : 'object',
                'properties': {
                    'critique': {
                        'type': 'string',
                        'description': 'A refinement of the research'
                    }
                },
                'required': ['critique']
            }
        }
    }



class Stage_1():
    def __init__(
        self,
        api_key            = API_KEY_REF,
        vlm_model          = "google/gemma-3-12b-it:free",
        provider_id        = "DeepInfra",
        model_temperature  = 0.7,
        max_tokens         = 2048,
        exp_task_name      = "carla_early",
        exp_task           = "Given a sequence of images shows a abstract robot walking, describe in detail how the robot is walking in terms of balance and how natural it looks.",
        exp_image_list_len = 2,
        exp_len            = 1,
        resume_idx         = 0,             # in case of interruption
        trail_folder       = '/data/hiho/raw-carla/extracted-jl-2025_08_26',
        trail_num          = 't-1'
    ):
        # --- model parameters ---
        self.api_key           = api_key
        self.vlm_model         = vlm_model
        self.provider_id       = provider_id
        self.model_temperature = model_temperature
        self.max_tokens        = max_tokens
        # --- experiment parameters ---
        self.exp_task_name     = exp_task_name
        self.exp_task          = exp_task
        self.exp_img_list_len  = exp_image_list_len
        self.exp_len           = exp_len
        self.resume_idx        = resume_idx
        self.trail_folder      = trail_folder

        self.output_file       = f'./ex4-carla/{trail_num}/sa_s1_{self.exp_task_name}.json'
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))


    def predict(self):
        meta_data = {
            'task_name'         : self.exp_task_name,
            'image_list_length' : self.exp_img_list_len,
            'total_epochs'      : self.exp_len,
            'data_folder'       : self.trail_folder,
            'stage_1' : {
                'task'  : self.exp_task,
                'model' : {
                    'name'       : self.vlm_model,
                    'temperature': self.model_temperature,
                    'max_tokens' : self.max_tokens
                },
            }
        }

        # --- get the annotation data ---
        with open(f'{self.trail_folder}/annotations.json', 'rb') as f:
            annotation_data = json.load(f)
            f.close()
        if self.exp_len > len(annotation_data):
            cprint(f"Not enough annotation data available. Required: {self.exp_len}, Available: {len(annotation_data)}", 'red')
            sys.exit(1)
        if self.resume_idx > 0:
            for i in range(self.resume_idx):
                del annotation_data[f'r_{i}']


        # --- Process each image sequence pair and generate description
        results = {}
        total_len = self.exp_len-self.resume_idx
        # Parallel(n_jobs=4, backend='multiprocessing')(
        #     delayed(self.pair_sequence_describe)(round_id, round_data)
        #     for round_id, round_data in tqdm(islice(annotation_data.items(), total_len), desc="Processing sequences", total=total_len)
        # )

        for round_id, round_data in tqdm(islice(annotation_data.items(), total_len), desc="Processing sequences", total=total_len):
            self.pair_sequence_describe(round_id, round_data)

        # --- Save to output file ---
        # find the saved temp round_result in the output doc dictionary
        result_file_prefix = self.output_file.replace('.json', '_round_result_')
        temp_round_results_list = [entry.path for entry in os.scandir(os.path.dirname(self.output_file)) if entry.path.startswith(result_file_prefix)]

        for path in temp_round_results_list:
            with open(path, 'r') as f:
                temp_round_result = json.load(f)
                round_id = temp_round_result['round_id']
                results[round_id] = {
                    **annotation_data[round_id],
                    **temp_round_result['round_data']
                }
                f.close()

        with open(self.output_file, 'w') as f:
            json.dump({
                'meta_data': meta_data,
                'results': results
            }, f, indent=4)
            f.close()
        cprint(f"Descriptions saved to {self.output_file}", 'green')


    def chat(self, messages, response_format=ResponseFormat.critic):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            'model'       : self.vlm_model,
            'provider_id' : self.provider_id,
            'messages'    : messages,
            'response_format': response_format,
            'max_tokens'  : self.max_tokens,
            'temperature' : self.model_temperature,
        }
        response = requests.post(url, headers=headers, json=payload)
        try:
            response_content = response.json()['choices'][0]['message']['content']
            token_usage      = response.json()['usage']
            return {'response': response_content, 'token_usage': token_usage}
        except Exception as e:
            cprint(f"Error in response: {response.text}", 'red')
            sys.exit(f"API error: {str(e)}")


    # def parse_response(self, agent_type: str, raw_response: str):
    #     try:
    #         json_start  = raw_response.index('{')
    #         json_end    = raw_response.index('}') + 1
    #         json_str    = raw_response[json_start:json_end]
    #         json_str    = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
    #         match agent_type:
    #             # case 'planner':
    #             #     parsed_response = PlannerOutput.model_validate(json.loads(json_str))
    #             # case 'researcher':
    #             #     parsed_response = ResearcherOutput.model_validate(json.loads(json_str))
    #             case 'critic':
    #                 parsed_response = CriticOutput.model_validate(json.loads(json_str))
    #             case _:
    #                 cprint(f"Unknown agent type: {agent_type}", 'red')
    #                 sys.exit(1)
    #         return parsed_response.model_dump()
    #     except Exception as e:
    #         cprint(f"Error parsing response for {agent_type}: {str(e)}", 'red')
    #         cprint(f"Raw response: {raw_response}", 'red')
    #         response_error_log_url = self.output_file.replace('.json', f'_{agent_type}_error_log.json')
    #         if not os.path.exists(response_error_log_url):
    #             with open(response_error_log_url, 'w') as f:
    #                 f.write(raw_response)
    #                 f.close()
    #         else:
    #             with open(response_error_log_url, 'a') as f:
    #                 f.write(raw_response)
    #                 f.close()
    #         return {'error': raw_response, 'message': str(e)}


    # ------------------ Multi Agent System ------------------
    def save_temp_result(self, agent_type: str, result: dict, round_id=None, side=None):
        match agent_type:
            case 'planner':
                temp_result_file = self.output_file.replace('.json', f'_{agent_type}.json')

            case 'critic' | 'round_result':
                temp_result_file = self.output_file.replace('.json', f'_{agent_type}_{round_id}.json')
               
            case 'researcher':
                temp_result_file = self.output_file.replace('.json', f'_{agent_type}_{round_id}_{side}.json')

            case _:
                cprint(f"Unknown agent type: {agent_type}", 'red')
                sys.exit(1)

        with open(temp_result_file, 'w') as f:
            json.dump(result, f, indent=4)
            cprint(f"Temporary results saved to {temp_result_file}", 'green')
            f.close()


    #  ------------------ Helper functions ------------------ 
    def image_to_base64(self, image_path):
        with Image.open(image_path) as img:
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
                img_url = f"data:image/png;base64,{base64_img}"
            return img_url


    def get_reduced_img_sequence(self, img_folder_path):
        """Get a reduced sequence of images from the folder."""
        image_paths_list = [os.path.join(img_folder_path, img) for img in sorted(os.listdir(img_folder_path)) if img.endswith(('.png', '.jpg', '.jpeg'))]
        result_path_list = []
        if len(image_paths_list) > self.exp_img_list_len:
            step = len(image_paths_list) // self.exp_img_list_len
            for i in range(0, len(image_paths_list), step):
                result_path_list.append(image_paths_list[i])
                if len(result_path_list) >= self.exp_img_list_len:
                    break
        else:
            result_path_list = image_paths_list
        cprint(f"Reduced image sequence to {len(result_path_list)} images from {len(image_paths_list)} original images.", 'green')
        return result_path_list


    #  ------------------ controling functions ------------------ 
    def single_sequence_describe(self, img_folder_path):
        # ---- prepare the image parts ----
        image_path_list = self.get_reduced_img_sequence(img_folder_path)
        image_parts = []
        # Convert all images to OpenAI vision format
        for img_path in image_path_list:
            image_parts.append({
                "type"      : "image_url",
                "image_url" : {
                    "url": self.image_to_base64(img_path)
                }
            })
        
        
        # parser = PydanticOutputParser(pydantic_object=CriticOutput)
        # format_instructions = parser.get_format_instructions()
        messages = [
            # {"role": "system", "content": format_instructions},
            {"role": "user", "content": [
                {"type": "text", "text": self.exp_task},
                *image_parts,
            ]},
        ]

        total_prompt_tokens     = 0
        total_completion_tokens = 0

        critic_result = self.chat(messages)
        # critic_result['response'] = self.parse_response('critic', critic_result['response'])
        print(critic_result['response'])
        critic_result['response'] = json.loads(critic_result['response'])
        total_prompt_tokens     += critic_result['token_usage']['prompt_tokens']
        total_completion_tokens += critic_result['token_usage']['completion_tokens']

        return {
            "final_output"            : critic_result['response']['critique'],
            "total_prompt_tokens"     : total_prompt_tokens,
            "total_completion_tokens" : total_completion_tokens
        }


    def pair_sequence_describe(self, round_id, round_data):
        epoch_left_folder  = f'{self.trail_folder}/{round_data["round_name"]}__{round_data["left"]}'
        epoch_right_folder = f'{self.trail_folder}/{round_data["round_name"]}__{round_data["right"]}'

        left_description   = self.single_sequence_describe(epoch_left_folder)
        right_description  = self.single_sequence_describe(epoch_right_folder)

        round_result = {
            "round_id": round_id,
            "round_data": {
                "left_description"          : left_description["final_output"],
                "right_description"         : right_description["final_output"],
                'round_tk_usage': {
                    "token_usage_left": {
                        "prompt_tokens"     : left_description["total_prompt_tokens"],
                        "completion_tokens" : left_description["total_completion_tokens"],
                    },
                    "token_usage_right": {
                        "prompt_tokens"     : right_description["total_prompt_tokens"],
                        "completion_tokens" : right_description["total_completion_tokens"],
                    }
                }
            }
        }
        self.save_temp_result('round_result', round_result, round_id)
        return round_result 
