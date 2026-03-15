import logging
import os
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    current_file_path = os.path.abspath(__file__)
    return Path(current_file_path).parent.parent.parent


def dataset_cache_dir() -> Path:
    p = os.getenv("DATASET_CACHE_DIR")
    if (
        not p
        or (not (path := Path(p)).exists())
        or (path.is_dir() and os.access(path, os.W_OK))
    ):
        path = get_project_root() / "datasets"
        print(f"Falling back to '{path.as_uri()}' for DATASET_CACHE_DIR")
    return path

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

def get_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)