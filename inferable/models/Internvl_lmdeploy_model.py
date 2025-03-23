from inferable.models.base_model import BaseModel
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging
import torch
from transformers import AutoTokenizer
from inferable.models.prompt_utils import get_prompt_id, get_prompt_text
from inferable.models.utils import extract_json_info, align_keys, PREDICT_KEYS
from inferable.models.few_shot import get_few_shot
from inferable.models.utils import to_json
from itertools import islice
from io import BytesIO
import base64


#vllm imports
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig


logger = logging.getLogger(__name__)

class InternvlLmdeployModel(BaseModel):
    '''Using fast model with lmdeploy. 
    Mainly taken from https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/vl_pipeline.md#batch-prompts-inference
    and https://lmdeploy.readthedocs.io/en/latest/multi_modal/vl_pipeline.html#'''

    def __init__(self, model_name :str = "OpenGVLab/InternVL2-40B", prompt :str = "1", few_shot : str = "", key_alignment :bool = True, batched :bool = True) -> None:
        """Inits the InternvlLmdeployModel.

        Args:
            model_name (str, optional): The model to use. Defaults to "OpenGVLab/InternVL2-40B".
            prompt (str, optional): The prompt to use (can also be a number from the prompt util file). Defaults to "1".
            few_shot (str, optional): the few shot method to use. 
                The input should look like "5-static" or "3-similar" where the number is the number of few shot examples to get and the method is the few shot method to use. Defaults to "".
            key_alignment (bool, optional):If True, the keys in the returned json are aligned with the keys in the training data. Defaults to True.
            batched (bool, optional): If true then the images are processed in batched mode. Defaults to True.
        """
        self.predict_keys = PREDICT_KEYS
        self.model_name = model_name
        self.prompt = prompt
        self.few_shot = few_shot
        self.few_shot_method = get_few_shot(few_shot) if few_shot else None
        self.key_alignment = key_alignment
        self.batched = batched

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_data: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

        if self.few_shot_method:
            self.few_shot_method.fit(training_data, validation_data)


    def encode_image(self, image: Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
        number_gpus = torch.cuda.device_count()

        backend_config=TurbomindEngineConfig(
            tp=number_gpus,
            cache_max_entry_count=0.90, # increase that value if better performance is needed
            # session_len=16384 # check if needed (or increased for few shot methods) -> it defaults to None which should be fine!
        )
        gen_config = GenerationConfig(
            max_new_tokens=500, # usually around 300 tokens generated
            temperature=0.0,
            do_sample=False,
        )

        pipe = pipeline(self.model_name, backend_config=backend_config) # TODO: add model revision
        processed_prompt = get_prompt_text(self.prompt).replace('<image>\n', '')

        

        if self.batched:
            it = iter(test_data)
            while True:
                batch = list(islice(it, 40))
                if not batch:
                    return

                if self.few_shot_method:
                    batch_prompts = []
                    for image in batch:
                        messages = []
                        few_shot_list = self.few_shot_method.get_few_shot_examples(image)
                        # reverse list to get the closest examples at the end of the prompt, which is then followed by the prediction image
                        few_shot_list.reverse()
                        for few_shot_image, few_shot_dict in few_shot_list:
                            messages.append(self.get_user_message(few_shot_image))
                            messages.append(self.get_assistant_message(few_shot_dict))
                        messages.append(self.get_user_message(image))
                        batch_prompts.append(messages)
                    responses = pipe(batch_prompts, gen_config=gen_config)
                else:
                    inputs = [(processed_prompt, img) for img in batch]
                    responses = pipe(inputs, gen_config=gen_config)   

                for response in responses:
                    generated_text = response.text

                    return_dict = extract_json_info(generated_text)
                    if self.key_alignment:
                        return_dict = align_keys(self.predict_keys, return_dict)
                    return_dict['full_response'] = generated_text

                    yield return_dict
        else:
            for image in test_data:
                if self.few_shot_method:
                    messages = []
                    few_shot_list = self.few_shot_method.get_few_shot_examples(image)
                    # reverse list to get the closest examples at the end of the prompt, which is then followed by the prediction image
                    few_shot_list.reverse()
                    for few_shot_image, few_shot_dict in few_shot_list:
                        messages.append(self.get_user_message(few_shot_image))
                        messages.append(self.get_assistant_message(few_shot_dict))
                    messages.append(self.get_user_message(image))
                    response = pipe(messages, gen_config=gen_config)
                else:
                    response = pipe((processed_prompt, image), gen_config=gen_config)
                

                generated_text = response.text

                return_dict = extract_json_info(generated_text)
                if self.key_alignment:
                    return_dict = align_keys(self.predict_keys, return_dict)
                return_dict['full_response'] = generated_text

                yield return_dict
        

    def __str__(self):
        return "InternvlLmdeployModel_" + self.model_name.split("/")[-1] + \
            "_p-" + get_prompt_id(self.prompt) + '_fewshot-' + str(self.few_shot) + \
            "_batched-" + str(self.batched) + "_align-" + str(self.key_alignment)

# openai format prompt:
# https://lmdeploy.readthedocs.io/en/v0.4.2/inference/vl_pipeline.html
