from src.GeneralFunctions import finetune_model, test_model

if __name__ == "__main__":
    model_path = finetune_model()
    test_model(model_path)
