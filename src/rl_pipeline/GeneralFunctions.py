import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rl_pipeline.Constants import SAVE_PATH
from rl_pipeline.LlamaRLTrainer import LlamaRLTrainer
from rl_pipeline.RLConfig import RLConfig
from rl_pipeline.RougeScore import compare_slm_rouge_scores
from rl_pipeline.datasets.coqa import CoqaDataset


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
    tokenizer = AutoTokenizer.from_pretrained(path, token=hf_token) if hf_token else AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 token=hf_token) if hf_token else AutoModelForCausalLM.from_pretrained(
        path)
    return tokenizer, model


def finetune_model():
    config = RLConfig(
        num_epochs=1,
        learning_rate=1e-5,
        use_lora=True,
        num_responses=3,
        RLConfig="auto",
        hf_token=os.environ['hf_token']
    )
    trainer = LlamaRLTrainer(config)
    path_ = ("%s" % SAVE_PATH)
    dataset = CoqaDataset().load_dataset("train", 2)

    return trainer.train(
        dataset=dataset,
        system_prompt=f"You are given a paragraph, read and understand it and give answers for given question.",
        save_path=path_
    )


def test_model(path):
    ground_truth, new_responses, old_responses = responses(path)
    if ground_truth !=[] and new_responses !=[] and old_responses !=[]:
        print(compare_slm_rouge_scores(ground_truth, new_responses, old_responses))
    else:
        print("There exists a null value")


def responses(path):
    test_data = CoqaDataset().load_dataset(split="validation", no_of_records=2)
    new_responses = [query_model(path=path, question=data.prompt) for data in test_data]

    old_responses = [
        query_model(RLConfig.model_1b_path, question=data.prompt, hf_token=RLConfig.hf_token)
        for data in test_data]

    ground_truth = [data.expected_response for data in test_data]
    return ground_truth, new_responses, old_responses
