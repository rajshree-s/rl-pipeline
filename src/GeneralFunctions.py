import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.Constants import SAVE_PATH, DATASET
from src.LlamaRLTrainer import LlamaRLTrainer
from src.QuestionDataset import QuestionDataset
from src.RLConfig import RLConfig
from src.RougeScore import compare_slm_rouge_scores
from src.StructureDataset import make_dict, StructureDataset


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


def finetune_model():
    config = RLConfig(
        num_epochs=1,
        learning_rate=1e-5,
        use_lora=True,
        num_responses=3,
        hf_token=os.environ['hf_token']
    )
    trainer = LlamaRLTrainer(config)
    path_ = ("%s" % SAVE_PATH)
    questions_para_data = make_dict(StructureDataset(path=DATASET, start=0, end=1).get_questions_with_paragraph())
    dataset = QuestionDataset(questions_para_data)
    return trainer.train(
        dataset=dataset,
        system_prompt=f"You are given a paragraph, read and understand it and give answers for given question.",
        save_path=path_
    )


def test_model(path):
    ground_truth, new_responses, old_responses = responses(path)
    print(compare_slm_rouge_scores(ground_truth, new_responses, old_responses))


def responses(path):
    test_data = StructureDataset(DATASET, start=7000, end=7001).get_question_answer_pairs()

    if os.path.exists('new_responses.json'):
        with open('new_responses.json', 'r') as f:
            new_responses = json.load(f)
    else:
        new_responses = [query_model(path=path, question=data.question) for data in test_data]
        with open('new_responses.json', 'w') as f:
            f.write(json.dumps(new_responses))

    if os.path.exists('old_responses.json'):
        with open('old_responses.json', 'r') as f:
            old_responses = json.load(f)
    else:
        old_responses = [
            query_model(RLConfig.model_1b_path, question=data.question, hf_token=RLConfig.hf_token)
            for data in test_data]
        with open('old_responses.json', 'w') as f:
            f.write(json.dumps(old_responses))

    ground_truth = [data.answer for data in test_data]
    return ground_truth, new_responses, old_responses
