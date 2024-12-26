from inferable.models.base_model import BaseModel
from typing import Dict, Iterable, List
import datasets
from PIL.Image import Image
from inferable.models.utils import PREDICT_KEYS
#from itertools import islice
#from time import sleep

class DummyModel(BaseModel):
    
    def __init__(self, predict_keys : List[str] = PREDICT_KEYS) -> None:
        self.predict_keys = predict_keys
    
    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')
        self.predict_keys.append('text_produced')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:

        #it = iter(test_data)
        #while True:
        #    batch = list(islice(it, 500))            
        #    if not batch:
        #        return
        #    print(f"Predicting batch of size {len(batch)}")
        #    for image in batch:
        #        print(image.filename)
        #        yield { prediction_key: "" for prediction_key in self.predict_keys }
        #    sleep(0.1)

        for _ in test_data:
            yield { prediction_key: "" for prediction_key in self.predict_keys}
    
    def __str__(self):
        return "DummyModel"