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

        answer_iter = iter(dataset['answers'])

        return [
            DatasetEntry(
                prompt=q,
                system_prompt=f"You are expert in reading comprehension task and here is your para: {story}",
                expected_response=a['input_text']
            )
            for story, list_of_questions in zip(dataset['story'], dataset['questions'])
            for q, a in zip(list_of_questions, answer_iter)
        ]


@dataclass
class DatasetEntry:
    prompt: str
    system_prompt: str
    expected_response: str
