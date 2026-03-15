from unittest import TestCase

from transformers import AutoTokenizer, AutoModelForCausalLM

from rl_pipeline.GeneralFunctions import get_response_for
from rl_pipeline.RLConfig import RLConfig


class TestFunctionsFor1BModel(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RLConfig.model_1b_path, token=RLConfig.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(RLConfig.model_1b_path, token=RLConfig.hf_token)
        self.device = "mps"

    def test_get_response_accepts_given_params_and_responds_with_answer_when_given_a_model(self):
        current_question = "How did Rachel fell off the car?"
        system_prompt = "You are comprehension reader, who reads paragraph and answer questions"
        comprehension = (
            "Rachel and Ross where driving to New York and suddenly there came a cow in front of them, and Ross and to "
            "grip up the hand break. Rachel wasn't wearing a seat belt and so was thrown out of the car.")
        previously_answered_questions = "What happened to the car?"

        response = get_response_for(
            self.device,
            self.model,
            self.tokenizer,
            current_question,
            system_prompt,
            comprehension,
            previously_answered_questions
        )

        print(response)
        assert len(response) > 0

