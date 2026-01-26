import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rl_pipeline.Constants import SAVE_PATH
from rl_pipeline.LlamaRLTrainer import LlamaRLTrainer
from rl_pipeline.RLConfig import RLConfig
from rl_pipeline.RougeScore import compare_slm_rouge_scores
from rl_pipeline.datasets.coqa import CoqaDataset
import json
import os


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
        hf_token=os.environ['hf_token']
    )
    trainer = LlamaRLTrainer(config)
    path_ = ("%s" % SAVE_PATH)
    dataset = CoqaDataset().load_dataset("train", 2)
    print("Dataset loaded Successfully")

    return trainer.train(
        dataset=dataset,
        system_prompt=f"You are given a paragraph, read and understand it and give answers for given question.",
        save_path=path_
    )


def test_model(path):
    ground_truth, new_responses, old_responses = responses(path)
    print("testing_model")
    if ground_truth != [] and new_responses != [] and old_responses != []:
        print(compare_slm_rouge_scores(ground_truth, new_responses, old_responses))
    else:
        print("There exists a null value")


def save_list_to_file(data_list, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f)
        print(f"Successfully saved to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save: {e}")
        return False


def load_list_from_file(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded {len(data)} items.")
            return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def get_responses(test_data, path, filename):
    if not os.path.exists(filename):
        response = [
            query_model(path, question=data.system_prompt + data.prompt, hf_token=RLConfig.hf_token)
            for data in test_data]
        save_list_to_file(response, filename)
        return response
    print("using the saved responses")
    return load_list_from_file(filename)

def save_list_to_file(data_list, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f)
        print(f"Successfully saved to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save: {e}")
        return False

def responses(path):
    test_data = CoqaDataset().load_dataset(split="validation", no_of_records=2)
    print(f"Here is the test data: {test_data}")
    print("fetching the old responses")
    old_responses = get_responses(test_data, RLConfig.model_1b_path, "old_responses.json")
    print("fetching the new responses")
    new_responses = get_responses(test_data, path, "new_responses.json")

    ground_truth = [data.expected_response for data in test_data]
    return ground_truth, new_responses, old_responses
