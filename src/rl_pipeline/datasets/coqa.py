from dataclasses import dataclass

from datasets import Dataset as HFDataset
from pandas import DataFrame
from typing import cast

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
        ground_truths = [answer['input_text'] for answer in answer_iter]


        return [
            DatasetEntry(
                prompt=q,
                system_prompt=f"You are expert in reading comprehension task and here is your para: {story}",
                expected_response=a
            )
            for story, list_of_questions in zip(dataset['story'], dataset['questions'])
            for answers in ground_truths
            for q, a in zip(list_of_questions, answers)
        ]


@dataclass
class DatasetEntry:
    prompt: str
    system_prompt: str
    expected_response: str

def _transform_from_raw(self, dataset: HFDataset) -> HFDataset:
    df: DataFrame = cast(DataFrame, dataset.to_pandas())
    df["answer"] = df["answers"].map(lambda x: x.get("input_text"))
    df["system_prompt"] = df["story"].map(
        lambda x: f"You are an expert in reading comphrehension task and here is you paragraph: {x}"
    )
    df = df.explode(["questions", "answer"]).rename({"questions": "prompt"})

    return HFDataset.from_pandas(df)
