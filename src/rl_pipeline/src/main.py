import os

from rl_pipeline.datasets.coqa import CoqaDataset
from rl_pipeline.src.config import RLConfig
from rl_pipeline.src.constants import SAVE_PATH
from rl_pipeline.src.rl_trainer import LlamaRLTrainer
from rl_pipeline.src.utils import get_logger

logger = get_logger()

if __name__ ==  "__main__":
    config = RLConfig(
        num_epochs=1,
        learning_rate=1e-5,
        use_lora=True,
        num_responses=3,
        hf_token=os.environ['hf_token']
    )
    trainer = LlamaRLTrainer(config)
    path_ = ("%s" % SAVE_PATH)
    dataset = CoqaDataset().load_dataset("train", no_of_records=2)
    logger.info("Dataset loaded Successfully")
    model_path = trainer.train(
        dataset=dataset,
        system_prompt=f"You are given a paragraph, read and understand it and give answers for given question.",
        save_path=path_
    )