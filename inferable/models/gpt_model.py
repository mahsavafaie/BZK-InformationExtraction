from inferable.models.base_model import BaseModel
from typing import Dict, Iterable
import datasets
from PIL.Image import Image
import logging

from io import BytesIO
import base64
from openai import OpenAI

from inferable.models.prompt_utils import get_prompt_id, get_prompt_text
from inferable.models.utils import extract_json_info, align_keys, PREDICT_KEYS
from inferable.models.few_shot import get_few_shot
from inferable.models.utils import to_json

logger = logging.getLogger(__name__)

class GPTModel(BaseModel):
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18", prompt: str = "1", few_shot : str = "", response_format: str = None, key_alignment :bool = True) -> None:
        """Inits the GPT Model.

        Args:
            model (str, optional): Choose a OpenAI model link to model: https://platform.openai.com/docs/models/gp#current-model-aliases. Defaults to "gpt-4o-mini-2024-07-18".
            prompt (str, optional): the prompt to use. Can also be an alias for a prompt defined in prompt_utils.py. Defaults to "1".
            few_shot (str, optional): the few shot method to use. 
                The input should look like "5-static" or "3-similar" where the number is the number of few shot examples to get and the method is the few shot method to use. Defaults to "".
            response_format (str, optional): if None, no response format is used. If 'json', at least a json is returned but schema can be arbitrary.
                If `structured` then the returned json adheres to the schema. See https://platform.openai.com/docs/guides/structured-outputs#structured-outputs-vs-json-mode .
                Defaults to None.
            key_alignment (bool, optional): If True, the keys in the returned json are aligned with the keys in the training data. Defaults to True.
        """
        self.predict_keys = PREDICT_KEYS
        self.model = model
        self.prompt = prompt
        self.response_format = response_format
        self.key_alignment = key_alignment
        self.few_shot = few_shot
        self.few_shot_method = get_few_shot(few_shot) if few_shot else None

        self.client = OpenAI()

    def encode_image(self, image: Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

        if self.few_shot_method:
            self.few_shot_method.fit(training_data, validation_dat)

    def get_response_format(self):
        if self.response_format is None:
            return None
        elif self.response_format == 'json':
            return {"type": "json_object"}
        elif self.response_format == 'structured':
            return {
                "type": "json_schema", 
                "json_schema": {
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": { key: {"type": "string"} for key in self.predict_keys },
                        "required": self.predict_keys,
                        "additionalProperties": False,
                    }
                }
            }
        else:
            raise ValueError(f"Invalid response format: {self.response_format}")


    def get_user_message(self, image: Image):
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": get_prompt_text(self.prompt).replace('<image>\n', '').replace('<image>', ''),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                    },
                },
            ],
        }
    def get_assistant_message(self, metadata: Dict[str, str]):
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": to_json(metadata, self.predict_keys, keep_empty_columns=True),
                }
            ]
        }
    

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        # https://platform.openai.com/docs/guides/vision?lang=curl#uploading-base64-encoded-images        

        for prediction_image in test_data:
            messages = []
            if self.few_shot_method:
                few_shot_list = self.few_shot_method.get_few_shot_examples(prediction_image)
                # reverse list to get the closest examples at the end of the prompt, which is then followed by the prediction image
                few_shot_list.reverse()
                for few_shot_image, few_shot_dict in few_shot_list:
                    messages.append(self.get_user_message(few_shot_image))
                    messages.append(self.get_assistant_message(few_shot_dict))

            messages.append(self.get_user_message(prediction_image))

            response = self.client.chat.completions.create(
                model=self.model,
                # https://community.openai.com/t/why-the-api-output-is-inconsistent-even-after-the-temperature-is-set-to-0/329541/12
                # https://community.openai.com/t/achieving-deterministic-api-output-on-language-models-howto/418318
                top_p=.0000000000000000000001, 
                messages=messages,
                response_format=self.get_response_format()
            )
            response_message = response.choices[0].message.content

            return_dict = extract_json_info(response_message)
            if self.key_alignment:
                return_dict = align_keys(self.predict_keys, return_dict)
            return_dict['full_response'] = response_message

            yield return_dict
    
    def __str__(self):
        return "GPTModel_" + self.model.split("/")[-1] + \
            '_p-' + get_prompt_id(self.prompt) + '_fewshot-' + str(self.few_shot) + \
            '_response-' + str(self.response_format) + "_align-" + str(self.key_alignment)

# order of multiple images
# https://community.openai.com/t/gpt4-v-the-order-of-multiple-image-inputs/519966/3