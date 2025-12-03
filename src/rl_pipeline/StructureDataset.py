import json
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class ParagraphQuestion:
    paragraph: str
    questions: List[str]


@dataclass
class QuestionAnswer:
    question: str
    answer: str


def make_dict(dataset: List[ParagraphQuestion | QuestionAnswer] = None):
    if dataset is None:
        return
    return [asdict(data) for data in dataset]


class StructureDataset:
    def __init__(self, path, start=None, end=None):
        with open(path, 'r') as f:
            self.data = json.load(f)
        if start or end:
            self.data = self.data[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_questions_with_paragraph(self):
        return [
            ParagraphQuestion(paragraph=data['story'], questions=data['questions']) for data in self.data
        ]

    def get_question_answer_pairs(self):
        questions= [ question for data in self.data for question in data['questions']]
        answers = [answer for data in self.data for answer in data['answers']]
        return [
            QuestionAnswer(question=question, answer=answer) for question, answer in zip(questions, answers)
        ]
