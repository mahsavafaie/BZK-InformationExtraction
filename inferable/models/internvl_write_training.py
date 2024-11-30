from inferable.models.base_model import BaseModel
from typing import Dict, Iterable
import datasets
from PIL.Image import Image
import os
import logging
import json
from inferable.models.utils import to_json, to_xml

logger = logging.getLogger(__name__)

class InternvlWriteTrainModel(BaseModel):
    """Write finetuning data according to the following webpages:
    - https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html
    - https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html
    """
    def __init__(self, root_folder: str, prompt: str="Extract BZK data", type: str="json", keep_empty_columns: bool= True) -> None:
        """Initializes the InternvlWriteTrainModel
        Args:
            root_folder (str): should point to /InternVL/internvl_chat when the git repository is cloned
            prompt (str, optional): the prompt to use for the user. Defaults to "Extract BZK data".
            type (str, optional): can either be json or xml (how the ground truth is formatted). Defaults to "json".
            keep_empty_columns (bool, optional): if true, then all columns are used independent if they are empty or not. Defaults to True.
        """
        self.root_folder = root_folder
        self.prompt = prompt
        self.type = type
        self.keep_empty_columns = keep_empty_columns
   
    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        ordered_dataset_keys = list(training_data.features.keys())
        ordered_dataset_keys.remove('image')

        # JSON data to be written to the file
        json_data = {
            "bzk_dataset": {
                "root": "playground/data/bzk", # TODO: change this
                "annotation": "playground/opensource/bzk.jsonl",# TODO: change this
                "data_augment": False,
                "repeat_time": 1,
                "length": training_data.num_rows
            }
        }
        os.makedirs(os.path.join(self.root_folder, "shell", "data"), exist_ok=True)
        with open(os.path.join(self.root_folder, "shell", "data", "internvl_1_2_finetune_custom.json"), "w") as f:
            json.dump(json_data, f, indent=4)
        
        if self.type.lower() == "json":
            serialize_function = to_json
        elif self.type.lower() == "xml":
            serialize_function = to_xml
        else:
            logger.error(f"Type {self.type} is not supported. Please use json or xml. Defaulting to json.")
            serialize_function = to_json
            
        # write jsonl file which hold all the data
        os.makedirs(os.path.join(self.root_folder, "playground", "opensource"), exist_ok=True)
        with open(os.path.join(self.root_folder, "playground", "opensource", "bzk.jsonl"), "w") as f:
            for i, example in enumerate(training_data):
                current_image = example["image"]
                width, height = current_image.size
                # save image to the folder
                os.makedirs(os.path.join(self.root_folder, "playground", "data", "bzk"), exist_ok=True)
                current_image.save(os.path.join(self.root_folder, "playground", "data", "bzk", f"image_{i:04}.jpg"))
                entry = {
                    "id": i,
                    "image": f"image_{i:04}.jpg",
                    "width": width,
                    "height": height,
                    "conversations": [ 
                        {
                            "from": "human",
                            "value" : "<image>\n" + self.prompt
                        },
                        {
                            "from": "gpt",
                            "value": serialize_function(example, ordered_dataset_keys, self.keep_empty_columns)
                        }
                    ]
                }

                json.dump(entry, f)
                f.write('\n')
        


    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        return []

    def __str__(self):
        return "InternvlWriteTrainModel"