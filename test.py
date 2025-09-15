import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()
HF_LLM_API_KEY = os.getenv("HF_LLM_API_KEY") 
client = InferenceClient(
    provider="nebius",
    api_key=HF_LLM_API_KEY,
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3-14B",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)