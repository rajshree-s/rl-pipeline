from typing import Dict, Any

from src.Constants import SAVE_PATH
from src.GeneralFunctions import query_saved_model
from src.LlamaRLTrainer import LlamaRLTrainer
from src.QuestionDataset import QuestionDataset
from src.RLConfig import RLConfig
from src.StructureDataset import FirstForm


def main():
    config = RLConfig(
        num_epochs=1,
        learning_rate=1e-5,
        use_lora=True,
        num_responses=3
    )
    trainer = LlamaRLTrainer(config)
    path_ = ("%s" % SAVE_PATH)
    questions_para_data = FirstForm("./dataset/cleaned_coqa_data.json")
    questions = questions_para_data.data[:10]

    # Loop the learning process
    train_and_save_model(path=path_, dataset=questions, trainer=trainer)


def train_and_save_model(path: str, dataset: Dict[Any, Any], trainer: LlamaRLTrainer):
    dataset = QuestionDataset(dataset)
    trainer.train(
        dataset=dataset,
        system_prompt=f"You are given a paragraph, read and understand it and give answers for given question.",
        save_path=path
    )


if __name__ == "__main__":
    main()
    path_ = ("%s" % SAVE_PATH)
    response = query_saved_model("./models/llama_1b_rl_trained_on_coqa_dataset_epoch_1_epoch_1")
    print(response)
