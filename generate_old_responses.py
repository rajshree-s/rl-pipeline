import sys

import torch

from rl_pipeline.GeneralFunctions import load_saved_model, save_list_to_file, load_list_from_file
from rl_pipeline.RLConfig import RLConfig
from rl_pipeline.datasets.coqa import CoqaDataset


def query_model(path, question: str, hf_token=None):
    print(f"Querying the model question: {question}")
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

    answer = response.strip()
    print(f"Here is the answer {answer}")
    print("------------------------------")
    return answer

def responses():
    test_data = CoqaDataset().load_dataset(split="validation", no_of_records=2)
    print(f"Here is the test data: {test_data}")
    old_responses = [query_model(path=RLConfig.model_1b_path, question=data.system_prompt + data.prompt) for data in test_data]
    filename = "old_responses.json"
    save_list_to_file(old_responses, filename)
    print("Saved new responses")
    print(f"Saved Response looks like: {load_list_from_file(filename)}")
    return old_responses

if __name__ == '__main__':
    responses()

