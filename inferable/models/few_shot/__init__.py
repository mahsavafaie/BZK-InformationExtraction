import importlib
from typing import List

few_shot_map = {
    "static": ("inferable.models.few_shot.static_few_shot", "StaticFewShot"),
    "similar": ("inferable.models.few_shot.similar_few_shot", "SimilarFewShot"),
    "similarcpu": ("inferable.models.few_shot.similarcpu_few_shot", "SimilarCPUFewShot"),
}


def get_few_shot_number(text: str):
    """Get the number of few shot examples from the text. The input should look like "5-static" or "5-similar" where the number is the number of few shot examples to get and the method is the few shot method to use.

    Args:
        text (str): the text to parse

    Returns:
        int: the number of few shot examples
    """
    if not text:
        return 0
    text = text.lower().strip()
    if not text:
        return 0
    
    # split text by minus
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError(f"Few shot argument {text} must contain a number and a few shot method e.g. 5-static")
    
    number, _ = parts
    #parse number to int and throw error if not possible
    try:
        parsed_number = int(number)
    except ValueError:
        raise ValueError(f"Number {number} could not be parsed to int")
    
    return parsed_number

def get_few_shot(text: str):
    """Create a few shot method object that can be used to get few shot examples for an image.
    The input should look like "5-static" or "5-similar" where the number is the number of few shot examples to get and the method is the few shot method to use.

    Args:
        text (str): the text to parse

    Returns:
        BaseFewShot: a few shot method
    """
    if not text:
        return None
    text = text.lower().strip()
    if not text:
        return None
    
    # split text by minus
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError(f"Few shot argument {text} must contain a number and a few shot method e.g. 5-static")
    
    number, few_shot_name = parts
    #parse number to int and throw error if not possible
    try:
        parsed_number = int(number)
    except ValueError:
        raise ValueError(f"Number {number} could not be parsed to int")
    
    if few_shot_name not in few_shot_map:
        raise ValueError(f"Few shot name {few_shot_name} not found in few_shot_map")
    module_name, class_name = few_shot_map[few_shot_name]
    my_class = getattr(importlib.import_module(module_name), class_name)
    return my_class(n=parsed_number)