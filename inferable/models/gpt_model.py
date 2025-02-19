from inferable.models.base_model import BaseModel
from typing import Dict, Iterable
import datasets
from PIL.Image import Image
import logging

from io import BytesIO
import base64
from openai import OpenAI

from inferable.models.prompt_utils import get_prompt_id, get_prompt_text
from inferable.models.utils import extract_json_info, align_keys

logger = logging.getLogger(__name__)

class GPTModel(BaseModel):
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18", prompt: str = "1", response_format: str = None, key_alignment :bool = True) -> None:
        """Inits the GPT Model.

        Args:
            model (str, optional): Choose a OpenAI model link to model: https://platform.openai.com/docs/models/gp#current-model-aliases. Defaults to "gpt-4o-mini-2024-07-18".
            prompt (str, optional): _description_. Defaults to "1".
            response_format (str, optional): if None, no response format is used. If 'json', at least a json is returned but schema can be arbitrary.
                If `structured` then the returned json adheres to the schema. See https://platform.openai.com/docs/guides/structured-outputs#structured-outputs-vs-json-mode .
                Defaults to None.
            key_alignment (bool, optional): If True, the keys in the returned json are aligned with the keys in the training data. Defaults to True.
        """
        self.predict_keys = None
        self.model = model
        self.prompt = prompt
        self.response_format = response_format
        self.key_alignment = key_alignment

        self.client = OpenAI()

    def encode_image(self, image: Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

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
                        #TODO: add schema
                    }
                }
            }
        else:
            raise ValueError(f"Invalid response format: {self.response_format}")

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        # https://platform.openai.com/docs/guides/vision?lang=curl#uploading-base64-encoded-images
        for image in test_data:
            base64_image = self.encode_image(image)
            response = self.client.chat.completions.create(
                model=self.model,
                # https://community.openai.com/t/why-the-api-output-is-inconsistent-even-after-the-temperature-is-set-to-0/329541/12
                # https://community.openai.com/t/achieving-deterministic-api-output-on-language-models-howto/418318
                top_p=.0000000000000000000001, 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": get_prompt_text(self.prompt).replace('<image>\n', '').replace('<image>', ''),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                response_format=self.get_response_format()
            )
            response_message = response.choices[0].message.content

            return_dict = extract_json_info(response_message)
            if self.key_alignment:
                return_dict = align_keys(self.predict_keys, return_dict)
            return_dict['full_response'] = response_message

            yield return_dict
    
    def __str__(self):
        return "GPTModel_" + self.model.split("/")[-1] + '_p-' + get_prompt_id(self.prompt) + '_resp-' + str(self.response_format) + "_ka-" + str(self.key_alignment)