from typing import Dict, List, Tuple
from datasets.arrow_dataset import Dataset
from PIL.Image import Image
from inferable.models.few_shot.base_few_shot import BaseFewShot

class StaticFewShot(BaseFewShot):

    def __init__(self) -> None:
        self.training_data = None

    def fit(self, training_data: Dataset, validation_dat: Dataset) -> None:
        self.training_data = training_data


    def get_few_shot_example(self, image : Image, n : int) -> List[Tuple[Image, Dict[str, str]]]:
        few_shot_examples = []
        for i in range(n):
            example = self.training_data[i]
            att_value = dict(example)
            example_img = att_value.pop('image')
            few_shot_examples.append((example_img, att_value))
        return few_shot_examples

    def get_few_shot_examples(self, images : List[Image], n : int) -> List[List[Tuple[Image, Dict[str, str]]]]:
        few_shot_examples = []
        for image in images:
            few_shot_examples.append(self.get_few_shot_example(image, n))
        return few_shot_examples