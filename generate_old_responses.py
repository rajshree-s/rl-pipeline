import json
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from bert_score import score
from rouge_score import rouge_scorer
from transformers import pipeline

from rl_pipeline.datasets.coqa import CoqaDataset

embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def query_model(pipe, system_prompt: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
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
    }


def calculate_rouge(model_answer, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = scorer.score(ground_truth, model_answer)

    return scores['rougeL'].fmeasure

def get_exact_embeddings_score(sentence1: str, sentence2:str):
    embedding1 = embeddings_model.encode(sentence1, convert_to_numpy=True)
    embedding2 = embeddings_model.encode(sentence2, convert_to_numpy=True)

    return abs(embedding2 - embedding1)


def responses() -> float:
    test_data = CoqaDataset().load_dataset(split="validation", no_of_records=10)

    total_score = 0
    similarity_score_rouge = 0
    count = 0

    pipe = load_pipe(model_id="meta-llama/Llama-3.2-1B-Instruct")
    with open("Insights.json", "w") as f:
        for data in test_data:
            prompt = f"{data.system_prompt} \n Here are the previously asked questions:{data.prev_context}\n"
            question = f"{data.prompt}"
            model_answer = query_model(
                pipe=pipe,
                system_prompt=prompt,
                question=question
            )
            ground_truth = data.expected_response
            # bert_score = get_bertscore_diff(model_answer, ground_truth)
            # similarity_score = bert_score['similarity_f1']
            # difference = bert_score['difference']
            # total_score += similarity_score
            # similarity_score_rouge += calculate_rouge(model_answer, ground_truth)
            embedding_score = get_exact_embeddings_score(model_answer, ground_truth)
            total_score +=embedding_score
            count += 1
            # json.dump({"Question no": count, "difference": difference}, f, indent=4)
            json.dump({"Question no": count, "Embedding Score":embedding_score}, f, indent=4)
        return total_score / count


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
    with open("output1.json", "w") as f:
        json.dump(output_data, f, indent=4)
