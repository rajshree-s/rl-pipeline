import json


class StructureDataset:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FirstForm(StructureDataset):
    def __init__(self, path):
        super().__init__(path)
        self.data = self._get_questions_with_paragraph()

    def _get_questions_with_paragraph(self):
        return [
            {
                "paragraph": data['story'],
                "questions": data['questions']
            }
            for data in self.data
        ]


