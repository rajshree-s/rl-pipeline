import json
from typing import List, Dict

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from rl_pipeline.Constants import SAVE_PATH
from datasets import Dataset
from rl_pipeline.RLConfig import RLConfig


class LlamaRLTrainer:
    """RL Trainer with sequential model loading for memory efficiency"""

    def __init__(self, config: RLConfig):
        self.config = config

        print("Loading tokenizers...")
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

        print("Loading 1B model (trainable)...")
        self.model_1b = AutoModelForCausalLM.from_pretrained(
            config.model_1b_path,
            torch_dtype=torch.float16,
            token=config.hf_token,
        )

        print("Loading 3B model (teacher)...")
        self.model_8b = AutoModelForCausalLM.from_pretrained(
            config.model_8b_path,
            torch_dtype=torch.float16,
            token=config.hf_token,
        )
        self.model_8b = self.model_8b.to(device=RLConfig.device)

        # Add LoRA to 1B for training
        if config.use_lora:
            print("Adding "
                  "LoRA to 1B model...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            self.model_1b = get_peft_model(self.model_1b, lora_config)
            self.model_1b = self.model_1b.to(device=RLConfig.device)
            print(f"Trainable params: {sum(p.numel() for p in self.model_1b.parameters() if p.requires_grad):,}")

        self.optimizer = torch.optim.AdamW(
            self.model_1b.parameters(),
            lr=config.learning_rate
        )

        print("Initialization complete!")

    def _load_reference_model(self):
        if self.model_8b is None:
            print("Loading reference 1B model...")
            self.model_8b = AutoModelForCausalLM.from_pretrained(
                self.config.model_8b_path,
                torch_dtype=torch.float16,
                token=self.config.hf_token
            )
            for param in self.model_8b.parameters():
                param.requires_grad = False

            self.model_8b = self.model_8b.to(device=RLConfig.device)
        return self.model_8b

    def _load_ranker_model(self):
        if self.model_8b is None:
            print("Loading ranker 1B model...")
            self.model_8b = AutoModelForCausalLM.from_pretrained(
                self.config.model_8b_path,
                torch_dtype=torch.float16,
                token=self.config.hf_token
            )
            for param in self.model_8b.parameters():
                param.requires_grad = False
            self.model_8b = self.model_8b.to(device=RLConfig.device)
        return self.model_8b

    def generate_responses(self, question: str, system_prompt: str, prompt: str, context: str) -> List[str]:
        """Generate responses from 1B model"""
        prompt = f"{system_prompt} \n\n Paragraph: {prompt} \n\nQuestion: {question}\n Here are the previously asked questions:{context}\n Answer:"

        inputs = self.tokenizer_1b(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)

        responses = []
        for _ in range(self.config.num_responses):
            with torch.no_grad():
                outputs = self.model_1b.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer_1b.pad_token_id
                )

            response = self.tokenizer_1b.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses

    def rank_responses(self, question: str, responses: List[str]) -> Dict[str, int]:
        """Use ranker model to rank responses"""
        ranker = self._load_ranker_model()

        ranking_prompt = f"""Rank these answers to the question from best (1) to worst ({len(responses)}).

            Question: {question}
            
            Answer 1: {responses[0]}
            Answer 2: {responses[1]}
            
            Provide rankings in JSON format: {{"answer1": <rank>, "answer2": <rank>}}"""

        inputs = self.tokenizer_8b(
            ranking_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)

        with torch.no_grad():
            outputs = ranker.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer_8b.pad_token_id
            )

        ranking_text = self.tokenizer_8b.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        # Parse rankings
        try:
            start = ranking_text.find('{')
            end = ranking_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = ranking_text[start:end]
                rankings = json.loads(json_str)
            else:
                rankings = {f"answer{i + 1}": i + 1 for i in range(len(responses))}
        except:
            rankings = {f"answer{i + 1}": i + 1 for i in range(len(responses))}

        return rankings

    def compute_rewards(self, rankings: Dict[str, int], num_responses: int) -> List[float]:
        """Convert rankings to rewards"""
        rewards = []
        for i in range(1, num_responses + 1):
            rank = rankings.get(f"answer{i}", i)
            reward = 1.0 - (rank - 1) / num_responses
            rewards.append(reward)
        return rewards

    def compute_ppo_loss(
            self,
            question: str,
            responses: List[str],
            rewards: torch.Tensor,
            system_prompt: str,
            prompt: str
    ) -> torch.Tensor:
        """Compute PPO loss for training"""
        ref_model = self._load_reference_model()
        prompt = f"{system_prompt}\n\n Paragraph: {prompt}\n\nQuestion: {question}\nAnswer:"

        total_loss = 0
        for idx, (response, reward) in enumerate(zip(responses, rewards)):
            full_text = prompt + response

            inputs = self.tokenizer_1b(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            outputs = self.model_1b(**inputs)
            logits = outputs.logits

            with torch.no_grad():
                ref_outputs = ref_model(**inputs)
                ref_logits = ref_outputs.logits

            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)

            target_tokens = inputs['input_ids'][:, 1:]
            selected_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
            ref_selected_log_probs = ref_log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

            ratio = torch.exp(selected_log_probs - ref_selected_log_probs)
            advantage = torch.tensor(reward, dtype=torch.float32, device=self.config.device)
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            )

            policy_loss = -torch.min(
                ratio * advantage,
                clipped_ratio * advantage
            ).mean()

            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
            loss = policy_loss - self.config.entropy_coef * entropy
            total_loss += loss

        return total_loss / len(responses)

    def train_step(self, question: str, system_prompt: str, prompt: str, prev_context: str):
        """Single training step"""
        responses = self.generate_responses(question, system_prompt, prompt, prev_context)
        rewards = self.reward_function(question, responses)

        self.optimizer.zero_grad()
        loss = self.compute_ppo_loss(question, responses, rewards, system_prompt, prompt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_1b.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), responses

    def reward_function(self, question: str, responses: list[str]) -> list[float]:
        rankings = self.rank_responses(question, responses)
        rewards = self.compute_rewards(rankings, len(responses))
        return rewards

    def train(self, dataset: Dataset, system_prompt: str, save_path: str = "%s" % SAVE_PATH):

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            total_loss = 0

            progress_bar = tqdm(dataset, desc="Training")
            for batch_idx, batch in enumerate(progress_bar):
                question = batch.prompt
                print(f'Here is the question: {question}')
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
                    print(f"Error on question: {question}")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            avg_loss = total_loss / max(len(dataset), 1)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
            model_path = f"{save_path}_epoch_{epoch + 1}"
            self.save_model(model_path)

        print("\nTraining complete!")
        return model_path

    def save_model(self, path: str):
        """Save the trained model"""
        self.model_1b.save_pretrained(path)
        self.tokenizer_1b.save_pretrained(path)
        print(f"Model saved to {path}")
