import json
import logging
import os
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rl_pipeline.Constants import SAVE_PATH
from rl_pipeline.LlamaRLTrainer import LlamaRLTrainer
from rl_pipeline.RLConfig import RLConfig
from rl_pipeline.RougeScore import compare_slm_rouge_scores
from rl_pipeline.datasets.coqa import CoqaDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    dataset = CoqaDataset().load_dataset("train")
    logger.info("Dataset loaded Successfully")

    return trainer.train(
        dataset=dataset,
        system_prompt=f"You are given a paragraph, read and understand it and give answers for given question.",
        save_path=path_
    )


def get_response_for(
        device_to_load: str,
        model: Any,
        tokenizer: Any,
        current_question: str,
        system_prompt: str,
        comprehension: str,
        previously_answered_questions: str
) -> str:
    chat_prompt = get_prompt_template(comprehension, current_question, previously_answered_questions, system_prompt,
                                      tokenizer)

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device_to_load)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_return_sequences=1
    )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response


def get_prompt_template(comprehension, current_question, previously_answered_questions, system_prompt, tokenizer):
    final_prompt = (f" \n\n Paragraph: {comprehension} \n\n Just for context you have answered previously"
                    f" these questions:{previously_answered_questions} \n Here is a new question: {current_question}\n"
                    f" Answer:")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_prompt


def test_model(path):
    ground_truth, new_responses, old_responses = responses(path)
    logger.info("testing_model")
    if ground_truth != [] and new_responses != [] and old_responses != []:
        logger.info(compare_slm_rouge_scores(ground_truth, new_responses, old_responses))
    else:
        logger.warning("There exists a null value in ground_truth, new_responses, or old_responses.")


def save_list_to_file(data_list, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f)
        logger.info(f"Successfully saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return False


def load_list_from_file(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} items from {filename}.")
            return data
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        return []


def get_responses(test_data, path, filename):
    if not os.path.exists(filename):
        response = [
            query_model(path, question=data.system_prompt + data.prompt, hf_token=RLConfig.hf_token)
            for data in test_data]
        save_list_to_file(response, filename)
        return response
    logger.info("using the saved responses")
    return load_list_from_file(filename)


def save_list_to_file(data_list, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f)
        logger.info(f"Successfully saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return False


def responses(path):
    test_data = CoqaDataset().load_dataset(split="validation", no_of_records=2)
    logger.info(f"Here is the test data: {test_data}")
    logger.info("fetching the old responses")
    old_responses = get_responses(test_data, RLConfig.model_1b_path, "old_responses.json")
    logger.info("fetching the new responses")
    new_responses = get_responses(test_data, path, "new_responses.json")

    ground_truth = [data.expected_response for data in test_data]
    return ground_truth, new_responses, old_responses
