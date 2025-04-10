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
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

logger = logging.getLogger(__name__)

class InternvlVllmModel(BaseModel):
    '''Using fast model with vllm. Mainly taken from https://docs.vllm.ai/en/v0.6.2/getting_started/examples/offline_inference_vision_language.html'''

    def __init__(self, model_name :str = "OpenGVLab/InternVL2-2B", prompt :str = "8", key_alignment :bool = True, batched :bool = True) -> None:
        self.predict_keys = PREDICT_KEYS
        self.model_name = model_name
        self.prompt = prompt
        self.key_alignment = key_alignment
        self.batched = batched

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_data: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        messages = [self.get_system_prompt_obj(), {'role': 'user', 'content': get_prompt_text(self.prompt)}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

        number_gpus = torch.cuda.device_count()
        llm = LLM(model=self.model_name,trust_remote_code=True, tensor_parallel_size=number_gpus,
                  gpu_memory_utilization=0.95, # default is 0.9 but to have more space for KV cache 
                  max_num_seqs=5, # max_num_seqs was 50
                  #enable_chunked_prefill=True, # results in error
                  #max_num_batched_tokens=256, # should be higher than the default of 512 for VLMs - based on https://github.com/vllm-project/vllm/issues/7996 
        ) #max_num_seqs=5,

        sampling_params = SamplingParams(temperature=0.0, # temperature=0.8, top_p=0.95
                                         max_tokens=500, # default is 16  - usually around 300 tokens generated
                                         stop_token_ids=stop_token_ids
        )
        # without chunked prefill Finished prediction in 318.4010114111006 seconds. for few files directory

        if self.batched:
            it = iter(test_data)
            while True:
                batch = list(islice(it, 50))            
                if not batch:
                    return
                
                inputs = [{
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img.convert("RGB"),
                    },
                } for img in batch]

                outputs = llm.generate(inputs, sampling_params)            
                for output in outputs:
                    generated_text = output.outputs[0].text

                    return_dict = extract_json_info(generated_text)
                    if self.key_alignment:
                        return_dict = align_keys(self.predict_keys, return_dict)
                    return_dict['full_response'] = generated_text

                    yield return_dict
        else:
            for img in test_data:
                inputs = {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img.convert("RGB"),
                    },
                }

                output = llm.generate(inputs, sampling_params)
                generated_text = output.outputs[0].text

                return_dict = extract_json_info(generated_text)
                if self.key_alignment:
                    return_dict = align_keys(self.predict_keys, return_dict)
                return_dict['full_response'] = generated_text

                yield return_dict

        # relevant urls:
        # https://github.com/vllm-project/vllm/issues/2492
        

    def get_system_prompt_obj(self) -> str:
        return {'role': 'system', 'content': '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'}
       

    def __str__(self):
        return "InternvlVLLMModel_" + self.model_name.split("/")[-1] + "_p" + get_prompt_id(self.prompt) + "_ka" + str(self.key_alignment) + "_b" + str(self.batched)

