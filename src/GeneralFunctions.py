from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.QuestionDataset import QuestionDataset


def query_model(path, question: str, hf_token=None):
    tokenizer, model = load_saved_model(path, hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=False,
            pad_token_id=model.config.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def load_saved_model(path: str, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(path, token= hf_token) if hf_token else AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, token= hf_token) if hf_token else AutoModelForCausalLM.from_pretrained(path)
    return tokenizer, model
