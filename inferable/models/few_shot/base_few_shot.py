from typing import Dict, List, Tuple
from datasets.arrow_dataset import Dataset
from abc import ABC, abstractmethod
from PIL.Image import Image

class BaseFewShot(ABC):
    
    @abstractmethod
    def fit(self, training_data: Dataset, validation_dat: Dataset) -> None:
        """Trains the model on the given training and validation data.
        Parameters
        ----------
        training_data: Dataset
            the dataset to train the model on (the image is stored in the 'image' column and all other columns are metadata)
        validation_dat: Dataset
            the dataset to validate the model on (the image is stored in the 'image' column and all other columns are metadata)
        Returns
        ----------"""
        pass
    
    @abstractmethod
    def get_few_shot_examples(self, image : Image) -> List[Tuple[Image, Dict[str, str]]]:
        """Gets a few shot example for the given image.
        Parameters
        ----------
        image: Image
            the image to get the few shot example for
        n: int
            the number of few shot examples to get
        Returns
        ----------
        List
            a list of tuples where the first element is the image and the second element is the metadata
        """
        pass
    
    @abstractmethod
    def get_multi_few_shot_examples(self, images : List[Image]) -> List[List[Tuple[Image, Dict[str, str]]]]:
        """Get few shot examples for multiple images. Might be faster than calling get_few_shot_example multiple times.

        Args:
            images (List[Image]): the list of images
            n (int): number of few shot examples to get

        Returns:
            List[List[Tuple[Image, Dict[str, str]]]]: a list of lists of tuples where the first element is the image and the second element is the metadata
        """
        pass