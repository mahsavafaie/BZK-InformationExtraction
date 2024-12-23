from inferable.models.base_model import BaseModel
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging

#vllm imports
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

logger = logging.getLogger(__name__)

class InternvlFastModel(BaseModel):
    '''Using fast model with vllm. Mainly taken from https://docs.vllm.ai/en/v0.6.2/getting_started/examples/offline_inference_vision_language.html'''

    def __init__(self, model_name :str = "OpenGVLab/InternVL2-Llama3-76B") -> None:
        self.model_name = model_name
        self.predict_keys = None

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:

        llm = LLM(
            model=self.model_name,
            #trust_remote_code=True,
            #max_num_seqs=5,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Stop tokens for InternVL
        # models variants may have different stop tokens
        # please refer to the model card for the correct "stop words":
        # https://huggingface.co/OpenGVLab/InternVL2-2B#service
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        return llm, prompt, stop_token_ids


        pass

    def __str__(self):
        return "InternvlFastModel"

