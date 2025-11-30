from src.Constants import SAVE_PATH
from src.LlamaRLTrainer import LlamaRLTrainer
from src.GeneralFunctions import query_saved_model
from src.QuestionDataset import QuestionDataset
from src.RLConfig import RLConfig



def main():
    config = RLConfig(
        num_epochs=1,
        learning_rate=1e-5,
        use_lora=True
    )

    system_prompt = "You are a helpful AI assistant. Provide clear, accurate answers."

    questions_data = [
        {"question": "What is machine learning?"},
        {"question": "Explain photosynthesis"},
        {"question": "How does a computer work?"}
    ]

    dataset = QuestionDataset(questions_data)
    trainer = LlamaRLTrainer(config)

    path_ = ("%s" % SAVE_PATH)
    trainer.train(
        dataset=dataset,
        system_prompt=system_prompt,
        save_path=path_
    )

    response = query_saved_model(path_)
    print(response)


if __name__ == "__main__":
    main()
