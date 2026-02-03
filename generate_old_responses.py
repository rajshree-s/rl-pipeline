from typing import Any

import json
import sys
import torch
from bert_score import score

from rl_pipeline.GeneralFunctions import load_saved_model, save_list_to_file, load_list_from_file
from rl_pipeline.RLConfig import RLConfig
from rl_pipeline.datasets.coqa import CoqaDataset
from transformers import pipeline


# def query_model(path, question: str, hf_token=None):
#     print(f"Querying the model question: {question}")
#     tokenizer, model = load_saved_model(path, hf_token)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id
#
#     inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512).to(model.device)
#
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             temperature=0.3,
#             do_sample=False,
#             pad_token_id=model.config.pad_token_id
#         )
#
#     response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#
#     answer = response.strip()
#     return answer


def query_model(system:str, user:str) -> str:
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    outputs = pipe(messages)
    return outputs[0]["generated_text"][-1]['content']


def get_bertscore_diff(candidate, reference):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)

    similarity = F1.item()

    difference = 1 - similarity

    return {
        "similarity_f1": round(similarity, 4),
        "difference": round(difference, 4)
    }['similarity_f1']


def responses() -> float:
    test_data = CoqaDataset().load_dataset(split="validation")
    similarity_score = 0
    count = 0
    for data in test_data:
        prompt = f"{data.system_prompt} \n Here are the previously asked questions:{data.prev_context}\n"
        question = f"{data.prompt}"
        model_answer = query_model(
            system=prompt,
            user=question
        )
        ground_truth = data.expected_response
        similarity_score += get_bertscore_diff(model_answer, ground_truth)
        count += 1
    return similarity_score / count


if __name__ == '__main__':
    score_value = responses()
    print(score_value)
    output_data = {"score": score_value}
    with open("output.json", "w") as f:
        json.dump(output_data, f, indent=4)
