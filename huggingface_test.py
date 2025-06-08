import os
from huggingface_hub import InferenceClient

with open('/home/sam/tokens/huggingface.txt', 'r') as file:  
    token = file.read().strip()

client = InferenceClient(
    provider="novita",
    api_key=token,
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
)

print(completion.choices[0].message)