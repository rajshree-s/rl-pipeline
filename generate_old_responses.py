import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from bert_score import score
from rouge_score import rouge_scorer
from transformers import pipeline

from rl_pipeline.datasets.coqa import CoqaDataset


def query_model(pipe, system:str, user:str) -> str:
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


def calculate_rouge(model_answer, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = scorer.score(ground_truth, model_answer)

    return scores['rougeL'].fmeasure


def responses() -> float:
    test_data = CoqaDataset().load_dataset(split="validation")

    similarity_score = 0
    similarity_score_rouge = 0
    count = 0

    pipe = load_pipe(model_id = "meta-llama/Llama-3.2-1B-Instruct")

    for data in test_data:
        prompt = f"{data.system_prompt} \n Here are the previously asked questions:{data.prev_context}\n"
        question = f"{data.prompt}"
        model_answer = query_model(
            pipe=pipe,
            system=prompt,
            user=question
        )
        ground_truth = data.expected_response
        similarity_score += get_bertscore_diff(model_answer, ground_truth)
        similarity_score_rouge += calculate_rouge(model_answer, ground_truth)
        count += 1
        logging.info("Similarity Score So far: %s", similarity_score/count)
        logging.info("Similarity Rouge Score So far: %s", similarity_score_rouge/count)
    return similarity_score / count


def load_pipe(model_id):
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


if __name__ == '__main__':
    score_value = responses()
    logging.info("Final score value: %s", score_value)
    output_data = {"score": score_value}
    with open("output.json", "w") as f:
        json.dump(output_data, f, indent=4)
