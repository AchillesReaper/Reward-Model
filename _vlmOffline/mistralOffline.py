from datetime import datetime, timedelta
import torch, base64

from PIL import Image
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download
from transformers import Mistral3ForConditionalGeneration

import numpy as np

import time
from termcolor import cprint

class MistralOffline():
    def __init__(
        self,
        model_id         = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        custom_cache_dir = "/home/hiho/Data/hf_cache_dir",
        max_new_tokens   = 1024,
        torch_dtype      = torch.bfloat16,
    ):
        self.custom_cache_dir = custom_cache_dir
        self.model_id         = model_id
        self.max_new_tokens   = max_new_tokens
        self.torch_dtype      = torch_dtype

        self.tokenizer  = MistralTokenizer.from_hf_hub(self.model_id)
        self.model      = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            cache_dir=self.custom_cache_dir
        )


    def chat(self, messages:list):
        tokenized      = self.tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
        input_ids      = torch.tensor([tokenized.tokens]).to(self.model.device)
        attention_mask = torch.ones_like(input_ids).to(self.model.device)

        # --- generate raw response ---
        if len(tokenized.images) > 0:       # for image inference
            pixel_values = torch.tensor(np.array(tokenized.images), dtype=self.torch_dtype).to(self.model.device)
            image_sizes  = torch.tensor([pv.shape[-2:] for pv in pixel_values]).to(self.model.device)
            output = self.model.generate(
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                pixel_values    = pixel_values,
                image_sizes     = image_sizes,
                max_new_tokens  = self.max_new_tokens,
                temperature     = 0.7,
            )[0]
        else:       # for text only inference
            output = self.model.generate(
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                max_new_tokens  = self.max_new_tokens,
                temperature     = 0.7,
            )[0]

        decoded_output = self.tokenizer.decode(output[len(tokenized.tokens) :])
        prompt_tokens = tokenized.tokens
        completion_tokens = output[len(tokenized.tokens) :]
        cprint(decoded_output, 'green')

        return {'response': decoded_output, 'token_usage':{
            'prompt_tokens': len(prompt_tokens),
            'completion_tokens': len(completion_tokens),
            'total_tokens': len(prompt_tokens) + len(completion_tokens)
        }}