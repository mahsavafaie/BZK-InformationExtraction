from inferable.models.base_model import BaseModel
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging

import torch
import math
import numpy as np
import torchvision.transforms as T
#from decord import VideoReader, cpu
import os
from torchvision.transforms.functional import InterpolationMode
from accelerate import init_empty_weights 
from transformers import AutoConfig, AutoTokenizer, AutoModel, BitsAndBytesConfig
from inferable.models.utils import extract_json_info, align_keys
from inferable.models.prompt_utils import get_prompt_id, get_prompt_text
from inferable.models.model_revisions import get_model_revision

logger = logging.getLogger(__name__)

class InternvlModel(BaseModel):
    """
    The InternvlModel is a model that uses InternVL2 from OpenGVLab to extract information from images.
    https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B
    Possible model names are:
    - OpenGVLab/InternVL2-40B
    - OpenGVLab/InternVL2-Llama3-76B
    """

    def __init__(self, model_name :str = "OpenGVLab/InternVL2-40B", prompt :str = "2", quantization :bool = False, key_alignment :bool = True) -> None:
        self.predict_keys = None
        self.model_name = model_name
        self.prompt = prompt
        self.quantization = quantization
        self.key_alignment = key_alignment

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image, input_size=448, max_num=12):
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        def split_model(model_name):
            device_map = {}
            world_size = torch.cuda.device_count()

            # find the number of layers in the model name ( based on https://huggingface.co/blog/accelerate-large-models)
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            with init_empty_weights():
                model = AutoModel.from_config(config, trust_remote_code=True)
            num_layers = len(model.language_model.model.layers)
            # Since the first GPU will be used for ViT, treat it as 0.25 (instead of half) of a GPU.
            # to following number needs to be adapted for different GPU sizes.
            first_gpu_ratio = 0.25 # how much (percentage points) of the first GPU is already occupied by the vision model
            num_layers_per_gpu = math.ceil(num_layers / (world_size - first_gpu_ratio))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1.0-first_gpu_ratio))
            #print(num_layers_per_gpu)
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
            device_map['vision_model'] = 0
            device_map['mlp1'] = 0
            device_map['language_model.model.tok_embeddings'] = 0
            device_map['language_model.model.embed_tokens'] = 0
            device_map['language_model.output'] = 0
            device_map['language_model.model.norm'] = 0
            device_map['language_model.lm_head'] = 0
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        
            return device_map


        device_map = split_model(self.model_name) 
        #device_map = 'auto'

        revision = get_model_revision(self.model_name)

        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True) if self.quantization else None, #in 4-bit provides irrelevant results
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            revision=revision,
            device_map=device_map).eval()
        #print(model.hf_device_map)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False, revision=revision)
        #load the model before the loop

        prompt_text = get_prompt_text(self.prompt)
        for image in test_data:
            #the inference code
            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=False)

            response, history = model.chat(tokenizer, pixel_values, prompt_text, generation_config, history=None, return_history=True)
            
            return_dict = extract_json_info(response)
            if self.key_alignment:
                return_dict = align_keys(self.predict_keys, return_dict)
            return_dict['full_response'] = response

            yield return_dict

    def __str__(self):
        return "InternvlModel_" + self.model_name.split("/")[-1] + "_p" + get_prompt_id(self.prompt) + "_ka" + str(self.key_alignment)


####################################################
# Info:
# Llama3-67 B  on 2 A100 80GB GPUs
# only if first GPU is treated as 0.6 instead of 0.5 (half a GPU) works
#device_map = {}
#device_map['vision_model'] = 0
#device_map['mlp1'] = 0
#device_map['language_model.model.tok_embeddings'] = 0
#device_map['language_model.model.embed_tokens'] = 0
#device_map['language_model.output'] = 0
#device_map['language_model.model.norm'] = 0
#device_map['language_model.lm_head'] = 0

# 32 don't work
# 33: GPU 0: 98% GPU 1: 91%
# 34: GPU 0: 93% GPU 1: 96% 
# 35: GPU 0: 95% GPU 1: 94% 
# 36: GPU 0: 97% GPU 1: 92%
# 37: GPU 0: 98% GPU 1: 90%
# 38: don't work

#split_point = 35
#for i in range(split_point):
#    device_map[f'language_model.model.layers.{i}'] = 0
#for i in range(split_point, 80):
#    device_map[f'language_model.model.layers.{i}'] = 1
#print(device_map) 


# finetune: 
# https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/internvl_chat_finetune.py#L816
# https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html


#