
from datetime import datetime, timedelta
import torch

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download
from transformers import Mistral3ForConditionalGeneration

import time
from termcolor import cprint

def model_inference(
        messages, 
        model,
        max_new_tokens=1000,
        torch_dtype=torch.bfloat16, 
):
    start_time = time.time()

    tokenized      = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
    input_ids      = torch.tensor([tokenized.tokens]).to(model.model.device)
    attention_mask = torch.ones_like(input_ids).to(model.model.device)

    # --- generate raw response ---
    if len(tokenized.images) > 0:       # for image inference
        pixel_values    = torch.tensor(tokenized.images[0], dtype=torch_dtype).unsqueeze(0).to(model.model.device)
        image_sizes     = torch.tensor([pixel_values.shape[-2:]]).to(model.model.device)
        output = model.generate(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            pixel_values    = pixel_values,
            image_sizes     = image_sizes,
            max_new_tokens  = max_new_tokens,
            temperature     = 0.7,
        )[0]
    else:       # for text only inference
        output = model.generate(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            max_new_tokens  = max_new_tokens,
            temperature     = 0.7,
        )[0]

    decoded_output = tokenizer.decode(output[len(tokenized.tokens) :])
    print(decoded_output)
    end_time = time.time()
    cprint(f"Time taken: {end_time - start_time:.2f} seconds", 'yellow')


if __name__ == "__main__": 
    # --- model related parameters ---
    model_id         = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    custom_cache_dir = "/home/hiho/Data/hf_cache_dir"
    torch_dtype      = torch.bfloat16

    # --- question related parameters ---
    image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in the picture?",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    # --- model and tokenizer configuration ---
    tokenizer = MistralTokenizer.from_hf_hub(model_id)

    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir   = custom_cache_dir,
        device_map  = "auto",
        torch_dtype = torch_dtype,
    )

    model_inference(messages, model, torch_dtype=torch_dtype)



# In this situation, you are playing a Pokémon game where your Pikachu (Level 42) is facing a wild Pidgey (Level 17). Here are the possible actions you can take and an analysis of each:

# 1. **FIGHT**:
#    - **Pros**: Pikachu is significantly higher level than the wild Pidgey, which suggests that it should be able to defeat Pidgey easily. This could be a good opportunity to gain experience points and possibly items or money.
#    - **Cons**: There is always a small risk of Pikachu fainting, especially if Pidgey has a powerful move or a status effect that could hinder Pikachu. However, given the large level difference, this risk is minimal.

# 2. **BAG**:
#    - **Pros**: You might have items in your bag that could help in this battle, such as Potions, Poké Balls, or Berries. Using an item could help you capture Pidgey or heal Pikachu if needed.
#    - **Cons**: Using items might not be necessary given the level difference. It could be more efficient to just fight and defeat Pidgey quickly.

# 3. **POKÉMON**:
#    - **Pros**: You might have another Pokémon in your party that is better suited for this battle or that you want to gain experience. Switching Pokémon could also be strategic if you want to train a lower-level Pokémon.
#    - **Cons**: Switching Pokémon might not be necessary since Pikachu is at a significant advantage. It could also waste time and potentially give Pidgey a turn to attack.

# 4. **RUN**:
#    - **Pros**: Running away could be a quick way to avoid the battle altogether. This might be useful if you are trying to conserve resources or if you are in a hurry to get to another location.
#    - **Cons**: Running away means you miss out on the experience points, items, or money that you could gain from defeating Pidgey. It also might not be the most efficient use of your time if you are trying to train your Pokémon.

# ### Recommendation:
# Given the significant level advantage, the best action to take is likely **FIGHT**. This will allow you to quickly defeat Pidgey and gain experience points for Pikachu. If you are concerned about Pikachu's health, you could use the **BAG** to heal Pikachu before or during the battle. Running away or switching Pokémon does not seem necessary in this situation.
