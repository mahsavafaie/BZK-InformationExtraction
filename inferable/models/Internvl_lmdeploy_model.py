from inferable.models.base_model import BaseModel
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging
import torch
from transformers import AutoTokenizer
from inferable.models.prompt_utils import get_prompt_id, get_prompt_text
from inferable.models.utils import extract_json_info, align_keys, PREDICT_KEYS
from itertools import islice

#vllm imports
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig


logger = logging.getLogger(__name__)

class InternvlLmdeployModel(BaseModel):
    '''Using fast model with lmdeploy. 
    Mainly taken from https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/vl_pipeline.md#batch-prompts-inference
    and https://lmdeploy.readthedocs.io/en/latest/multi_modal/vl_pipeline.html#'''

    def __init__(self, model_name :str = "OpenGVLab/InternVL2_5-38B", prompt :str = "4", key_alignment :bool = True, batched :bool = True) -> None:
        self.predict_keys = PREDICT_KEYS
        self.model_name = model_name
        self.prompt = prompt
        self.key_alignment = key_alignment
        self.batched = batched

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_data: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        number_gpus = torch.cuda.device_count()

        backend_config=TurbomindEngineConfig(
            tp=number_gpus,
            cache_max_entry_count=0.95
        )
        gen_config = GenerationConfig(
            max_new_tokens=500, # usually around 300 tokens generated
            temperature=0.0,
            do_sample=False,
        )

        pipe = pipeline(self.model_name, backend_config=backend_config)
        processed_prompt = get_prompt_text(self.prompt).replace('<image>\n', '')

        

        if self.batched:
            it = iter(test_data)
            while True:
                batch = list(islice(it, 40))
                if not batch:
                    return
                
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
                response = pipe((processed_prompt, image), gen_config=gen_config)
                generated_text = response.text

                return_dict = extract_json_info(generated_text)
                if self.key_alignment:
                    return_dict = align_keys(self.predict_keys, return_dict)
                return_dict['full_response'] = generated_text

                yield return_dict
        

    def __str__(self):
        return "InternvlLmdeployModel_" + self.model_name.split("/")[-1] + "_p" + get_prompt_id(self.prompt) + "_ka" + str(self.key_alignment) + "_b" + str(self.batched)

