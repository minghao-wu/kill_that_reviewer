# Kill That Reviewer

The rebuttal period for the ACL ARR 2024 February cycle has concluded. It is evident that many people are upset about the reviews they received. I am releasing this small project to fine-tune a language model that can either join you in cursing your reviewers or provide you with encouragement.

This project is created for entertainment purposes and is intended to be enjoyed as such. If for any reason it causes discomfort or you have concerns about the content, please feel free to reach out to me and I will take everything down immediately.

HAVE FUN !!!

## Disclaimers

1. This content are not intended to offend any individual or group. If any content is found to be offensive, it is unintentional and I apologize for any discomfort caused.

2. Please be advised that the content you are about to read may contain language that could be considered offensive or inappropriate for some audiences. Reader discretion is advised.

3. I would like to assure our readers that no reviewers were harmed during the development of this content.

## Dataset

I used `gpt-3.5-turbo` to create 1,000 examples of cursing and 1,000 examples of encouragement. Each example consists of multi-turn conversations. It cost me about 1 dollar.

The dataset is available at [here](https://huggingface.co/datasets/minghaowu/kill_that_reviewer).

For example,
```
[
    { "role": "system", "content": "You are an assistant encouraging the user who is cursing and complaining about the reviewers." }, 
    { "role": "user", "content": "I can't believe these damn reviewers! They always find something to nitpick and complain about." }, 
    { "role": "assistant", "content": "I understand it can be frustrating, but remember that feedback, positive or negative, can only help you improve." }, 
    { "role": "user", "content": "But it's so demoralizing! I put so much effort into my work, and they just tear it apart." }, 
    { "role": "assistant", "content": "I hear you. Just try to focus on the constructive criticism, and use it to grow and become even better." }, 
    { "role": "user", "content": "I guess you're right. I shouldn't let them get to me. I'll work on taking their feedback in stride." }, 
    { "role": "assistant", "content": "That's the spirit! Keep pushing forward and don't let anyone discourage you from reaching your goals." } 
]
```

## Models

* [minghaowu/phi-2-ktr](https://huggingface.co/minghaowu/phi-2-ktr)
* [minghaowu/Qwen1.5-1.8B-ktr](https://huggingface.co/minghaowu/Qwen1.5-1.8B-ktr)

## Inference

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "minghaowu/phi-2-ktr"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", torch_dtype=torch.bfloat16)

system_prompt = "You are an assistant cursing and complaining about the reviewers with the user."
user_prompt = input("### USER: ")
infer_prompt = f"### SYSTEM: {system_prompt} <|endoftext|>\n### USER: {user_prompt} <|endoftext|>\n### ASSISTANT:"
output = pipe(infer_prompt, do_sample=True, max_new_tokens=256)[0]["generated_text"]

out = f"### ASSISTANT:{output[len(infer_prompt):]}"
print(out)
```