import logging

import torch
from bert_score import score as bert_score_func
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from rl_pipeline.Constants import SAVE_PATH
from rl_pipeline.GeneralFunctions import get_response_for
from rl_pipeline.RLConfig import RLConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaRLTrainer:

    def __init__(self, config: RLConfig):
        self.config = config

        logger.info("Loading tokenizers...")
        self.tokenizer_1b = AutoTokenizer.from_pretrained(
            config.model_1b_path,
            token=config.hf_token
        )
        self.tokenizer_8b = AutoTokenizer.from_pretrained(
            config.model_8b_path,
            token=config.hf_token
        )

        if self.tokenizer_1b.pad_token is None:
            self.tokenizer_1b.pad_token = self.tokenizer_1b.eos_token

        if self.tokenizer_8b.pad_token is None:
            self.tokenizer_8b.pad_token = self.tokenizer_8b.eos_token

        logger.info("Loading 1B model (trainable)...")
        self.model_1b = AutoModelForCausalLM.from_pretrained(
            config.model_1b_path,
            torch_dtype=torch.float16,
            token=config.hf_token,
        )

        logger.info("Loading 8B model (teacher)...")
        self.model_8b = AutoModelForCausalLM.from_pretrained(
            config.model_8b_path,
            torch_dtype=torch.float16,
            token=config.hf_token,
        )
        self.model_8b = self.model_8b.to(device=RLConfig.device)

        if config.use_lora:
            logger.info("Adding LoRA to 1B model...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            self.model_1b = get_peft_model(self.model_1b, lora_config)
            self.model_1b = self.model_1b.to(device=RLConfig.device)
            logger.info(f"Trainable params: {sum(p.numel() for p in self.model_1b.parameters() if p.requires_grad):,}")

        self.optimizer = torch.optim.AdamW(
            self.model_1b.parameters(),
            lr=config.learning_rate
        )

        logger.info("Initialization complete!")

    def compute_reinforce_loss(
            self,
            question: str,
            slm_response: str,
            llm_response: str,
            system_prompt: str,
            prompt: str,
            prev_context: str
    ) -> torch.Tensor:
        student_response = slm_response
        teacher_response = llm_response
        _, _, f1 = bert_score_func([student_response], [teacher_response], lang="en", model_type='bert-base-uncased',
                                   device=self.config.device)
        reward = f1.squeeze()

        prompt_text = f"{system_prompt}\n\n Paragraph: {prompt}\n\nQuestion: {question}\nHere are the previously asked questions:{prev_context}\n Answer:"
        full_text = prompt_text + student_response

        inputs = self.tokenizer_1b(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        ).to(self.config.device)

        outputs = self.model_1b(**inputs)
        logits = outputs.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        prompt_tokens_ids = self.tokenizer_1b(prompt_text, return_tensors="pt", padding=True).to(self.config.device)[
            'input_ids']
        prompt_length = prompt_tokens_ids.shape[1]

        target_tokens = inputs['input_ids'][:, 1:]

        response_log_probs = log_probs[:, prompt_length - 1:, :]
        response_target_tokens = target_tokens[:, prompt_length - 1:]

        selected_log_probs = response_log_probs.gather(2, response_target_tokens.unsqueeze(-1)).squeeze(-1)
        loss = -selected_log_probs.sum() * torch.clamp(reward, min=0.0)
        return loss

    def train_step(self, question: str, system_prompt: str, prompt: str, prev_context: str):
        slm_response = get_response_for(
            device_to_load=RLConfig.device,
            model=self.model_1b,
            tokenizer=self.tokenizer_1b,
            current_question=question,
            system_prompt=system_prompt,
            comprehension=prompt,
            previously_answered_questions=prev_context
        )
        llm_response = get_response_for(
            device_to_load=RLConfig.device,
            model=self.model_8b,
            tokenizer=self.tokenizer_8b,
            current_question=question,
            system_prompt=system_prompt,
            comprehension=prompt,
            previously_answered_questions=prev_context
        )

        self.optimizer.zero_grad()

        loss = self.compute_reinforce_loss(
            question=question,
            slm_response=slm_response,
            llm_response=llm_response,
            system_prompt=system_prompt,
            prompt=prompt,
            prev_context=prev_context)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_1b.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), slm_response

    def train(self, dataset: Dataset, system_prompt: str, save_path: str = "%s" % SAVE_PATH):
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            total_loss = 0

            progress_bar = tqdm(dataset, desc="Training")
            for batch_idx, batch in enumerate(progress_bar):
                question = batch.prompt
                logger.info(f'Here is the question: {question}')
                para = batch.system_prompt
                prev_context = batch.prev_context
                try:
                    loss, responses = self.train_step(question, system_prompt, para, prev_context)
                    total_loss += loss

                    progress_bar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                    })
                except Exception as e:
                    logger.error(f"Error on question: {question}")
                    logger.exception(f"Error: {e}")
                    import traceback
                    traceback.print_exc()

            avg_loss = total_loss / max(len(dataset), 1)
            logger.info(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
            model_path = f"{save_path}_epoch_{epoch + 1}"
            self.save_model(model_path)

        logger.info("\nTraining complete!")
        return model_path

    def save_model(self, path: str):
        """Save the trained model"""
        self.model_1b.save_pretrained(path)
        self.tokenizer_1b.save_pretrained(path)
        logger.info(f"Model saved to {path}")
