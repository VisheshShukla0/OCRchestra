import torch
from transformers import pipeline


def create_pipe():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        model_kwargs={"temperature": 0.7,
                  "do_sample":True,
                              "top_p":0.9,
                              "top_k":50,},
    )
    return pipe

def ask( pipe, query):
    messages = [
    {"role": "system", "content": "You are a an expert chatbot who carefully follow user's instructions."},
    {"role": "user", "content": f"{query}"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    response = outputs[0]["generated_text"][-1]['content']
    return response