from dataclasses import dataclass

from datasets import Dataset as HFDataset

from .dataset import Dataset


class CoqaDataset(Dataset):
    _name = "stanfordnlp/coqa"
    _splits = ("train", "validation")
    _promt = ""

    @property
    def name(self) -> str:
        return CoqaDataset._name

    @property
    def splits(self) -> tuple[str, ...]:
        return CoqaDataset._splits

    @property
    def prompt(self) -> str:
        return CoqaDataset._promt

    def _transform_from_raw(self, dataset: HFDataset):
        results = []
        ground_truths = [answer['input_text'] for answer in dataset['answers']]
        for story, list_of_questions, answers in zip(dataset['story'], dataset['questions'], ground_truths):
            prev_questions = []
            for q, a in zip(list_of_questions, answers):
                entry = DatasetEntry(
                    prompt=q,
                    prev_context="\n".join(prev_questions),
                    system_prompt=f"You are expert in reading comprehension task and here is your para: {story}",
                    expected_response=a
                )
                results.append(entry)
                prev_questions.append(q)
        return results


@dataclass
class DatasetEntry:
    prompt: str
    prev_context: str
    system_prompt: str
    expected_response: str
