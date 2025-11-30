from dataclasses import dataclass


@dataclass
class RLConfig:
    """Configuration for RL training - optimized for Colab"""
    model_1b_path: str = "meta-llama/Llama-3.2-1B"
    model_8b_path: str = "meta-llama/Llama-3.2-1B"  # Same 1B model
    batch_size: int = 1
    learning_rate: float = 1e-5
    num_epochs: int = 1  # Reduce for testing
    max_length: int = 256  # Reduced from 512
    num_responses: int = 2  # Reduced from 3
    temperature: float = 0.8
    top_p: float = 0.9
    ppo_epochs: int = 1
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    device: str = "cpu"
    hf_token: str = None  # Will use Colab Secrets
    use_lora: bool = True
    use_8bit: bool = False  # Set to True if OOM errors
