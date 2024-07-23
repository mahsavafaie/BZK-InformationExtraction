from datasets import load_dataset
from inferable.data.base_dataset import BaseDataset

class BZKDataset(BaseDataset):

    def get_training_data(self):
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="train")
    
    def get_validation_data(self):
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="validation")
    
    def get_test_data(self):
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="test")

class BZKDatasetSmall(BaseDataset):

    def get_training_data(self):
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="train[:10%]")
    
    def get_validation_data(self):
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="validation[:10%]")
    
    def get_test_data(self):
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="test[:10%]")