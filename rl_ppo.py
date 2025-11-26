hf_token = ""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from huggingface_hub import login
login(new_session=True)

@dataclass
class RLConfig:
    """Configuration for RL training"""
    model_1b_path: str = "meta-llama/Llama-3.2-1B"
    model_8b_path: str = "meta-llama/Llama-3.2-1B"
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 3
    max_length: int = 512
    num_responses: int = 3
    temperature: float = 0.8
    top_p: float = 0.9
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token: str = ""


class QuestionDataset(Dataset):
    """Dataset for questions"""
    def __init__(self, questions_file: str):
        with open(questions_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LlamaRLTrainer:
    """RL Trainer for Llama 3.2 1B using 8B as ranker"""

    def __init__(self, config: RLConfig):
        self.config = config

        # Initialize tokenizers
        print("Loading tokenizers...")
        self.tokenizer_1b = AutoTokenizer.from_pretrained(config.model_1b_path, token=config.hf_token)
        self.tokenizer_8b = AutoTokenizer.from_pretrained(config.model_8b_path, token=config.hf_token)

        # Set padding tokens
        if self.tokenizer_1b.pad_token is None:
            self.tokenizer_1b.pad_token = self.tokenizer_1b.eos_token
        if self.tokenizer_8b.pad_token is None:
            self.tokenizer_8b.pad_token = self.tokenizer_8b.eos_token

        # Initialize models
        print("Loading 1B model (trainable)...")
        self.model_1b = AutoModelForCausalLM.from_pretrained(
            config.model_1b_path,
            torch_dtype=torch.float16,
            device_map=config.device,
            token=config.hf_token
        )

        print("Loading 8B model (ranker)...")
        self.model_8b = AutoModelForCausalLM.from_pretrained(
            config.model_8b_path,
            torch_dtype=torch.float16,
            device_map=config.device,
            token=config.hf_token
        )

        # Freeze 8B model (only used for ranking)
        for param in self.model_8b.parameters():
            param.requires_grad = False

        # Keep reference model for PPO (frozen copy of 1B)
        print("Creating reference model...")
        self.ref_model_1b = AutoModelForCausalLM.from_pretrained(
            config.model_1b_path,
            torch_dtype=torch.float16,
            device_map=config.device,
            token=config.hf_token
        )
        for param in self.ref_model_1b.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model_1b.parameters(),
            lr=config.learning_rate
        )

        print("Initialization complete!")

    def generate_responses(self, question: str, system_prompt: str) -> List[str]:
        """Generate 3 different responses from 1B model"""
        prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"

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
                    max_new_tokens=256,
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
        """Use 8B model to rank the responses"""
        ranking_prompt = f"""Rank the following three answers to the question from best (1) to worst (3).

                            Question: {question}

                            Answer 1: {responses[0]}

                            Answer 2: {responses[1]}

                            Answer 3: {responses[2]}

                            Provide rankings in JSON format:
                            {{
                                "answer1": <rank>,
                                "answer2": <rank>,
                                "answer3": <rank>
                            }}"""

        inputs = self.tokenizer_8b(
            ranking_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model_8b.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer_8b.pad_token_id
            )

        ranking_text = self.tokenizer_8b.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Parse JSON rankings
        try:
            # Extract JSON from response
            start = ranking_text.find('{')
            end = ranking_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = ranking_text[start:end]
                rankings = json.loads(json_str)
            else:
                # Fallback: assign default rankings
                rankings = {"answer1": 2, "answer2": 2, "answer3": 2}
        except:
            # Fallback rankings if parsing fails
            rankings = {"answer1": 2, "answer2": 2, "answer3": 2}

        return rankings

    def compute_rewards(self, rankings: Dict[str, int]) -> torch.Tensor:
        """Convert rankings to rewards (rank 1 = best, rank 3 = worst)"""
        rewards = []
        for i in range(1, 4):
            rank = rankings[f"answer{i}"]
            # Convert rank to reward: rank 1 -> 1.0, rank 2 -> 0.0, rank 3 -> -1.0
            reward = 1.0 - (rank - 1)
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def compute_ppo_loss(
        self,
        question: str,
        responses: List[str],
        rewards: torch.Tensor,
        system_prompt: str
    ) -> torch.Tensor:
        """Compute PPO loss for training"""
        prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"

        total_loss = 0
        for idx, (response, reward) in enumerate(zip(responses, rewards)):
            full_text = prompt + response

            # Tokenize
            inputs = self.tokenizer_1b(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            # Get logits from current model
            outputs = self.model_1b(**inputs)
            logits = outputs.logits

            # Get logits from reference model
            with torch.no_grad():
                ref_outputs = self.ref_model_1b(**inputs)
                ref_logits = ref_outputs.logits

            # Compute log probabilities
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)

            # Get actual token log probs
            target_tokens = inputs['input_ids'][:, 1:]
            selected_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
            ref_selected_log_probs = ref_log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

            # Compute ratio for PPO
            ratio = torch.exp(selected_log_probs - ref_selected_log_probs)

            # Clipped surrogate objective
            advantage = reward.to(self.config.device)
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            )

            policy_loss = -torch.min(
                ratio * advantage,
                clipped_ratio * advantage
            ).mean()

            # Entropy bonus for exploration
            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

            loss = policy_loss - self.config.entropy_coef * entropy
            total_loss += loss

        return total_loss / len(responses)

    def train_step(
        self,
        question: str,
        system_prompt: str
    ) -> Tuple[float, Dict[str, int]]:
        """Single training step"""
        # Generate responses from 1B
        responses = self.generate_responses(question, system_prompt)

        # Get rankings from 3B
        rankings = self.rank_responses(question, responses)

        # Compute rewards
        rewards = self.compute_rewards(rankings)

        # PPO training
        self.optimizer.zero_grad()
        loss = self.compute_ppo_loss(question, responses, rewards, system_prompt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_1b.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), rankings

    def train(
        self,
        dataset: QuestionDataset,
        system_prompt: str,
        save_path: str = "./trained_model"
    ):
        """Main training loop"""
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Process one question at a time
            shuffle=True
        )

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            total_loss = 0

            progress_bar = tqdm(dataloader, desc=f"Training")
            for batch_idx, batch in enumerate(progress_bar):
                question = batch['question'][0]  # Unpack from batch

                loss, rankings = self.train_step(question, system_prompt)
                total_loss += loss

                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })

                # Log rankings every 10 steps
                if batch_idx % 10 == 0:
                    print(f"\nQuestion: {question[:100]}...")
                    print(f"Rankings: {rankings}")

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            checkpoint_path = f"{save_path}_epoch_{epoch+1}"
            self.save_model(checkpoint_path)

        print("\nTraining complete!")

    def save_model(self, path: str):
        """Save the trained 1B model"""
        self.model_1b.save_pretrained(path)
        self.tokenizer_1b.save_pretrained(path)
        print(f"Model saved to {path}")

def main():
    # Configuration
    config = RLConfig(
        batch_size=4,
        learning_rate=1e-5,
        num_epochs=3,
        hf_token=hf_token # Pass the token to the config
    )

    # System prompt for the model
    system_prompt = """You are a helpful AI assistant. Provide clear, accurate, and concise answers to questions."""

    # Load dataset (expects JSON file with list of dicts containing 'question' field)
    # Example format: [{"question": "What is Python?"}, {"question": "Explain ML"}]
    questions_data = [
    {"question": "What is machine learning?"},
    {"question": "Explain photosynthesis"},
    {"question": "How does a computer work?"}
    ]
    with open("questions.json", "w") as f:
        json.dump(questions_data, f)

    dataset = QuestionDataset("questions.json")

    # Initialize trainer
    trainer = LlamaRLTrainer(config)

    # Train
    trainer.train(
        dataset=dataset,
        system_prompt=system_prompt,
        save_path="./llama_1b_rl_trained"
    )

if __name__ == "__main__":
    main()