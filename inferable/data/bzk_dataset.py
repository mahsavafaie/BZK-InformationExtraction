from datasets import load_dataset
from inferable.data.base_dataset import BaseDataset
from datasets.arrow_dataset import Dataset

class BZKDatasetLocal(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="train")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="validation")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="test")

class BZKDatasetSmallLocal(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="train[:10%]")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="validation[:10%]")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="test[:10%]")
    

# huggingface datasets

class BZKDatasetNormalizedHF(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "normalized", split="train")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "normalized", split="validation")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "normalized", split="test")

class BZKDatasetRawHF(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "raw", split="train")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "raw", split="validation")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "raw", split="test")

class BZKDatasetSmallNormalizedHF(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "normalized", split="train[:10%]")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "normalized", split="validation[:10%]")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "normalized", split="test[:10%]")
    

class BZKDatasetSmallRawHF(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "raw", split="train[:10%]")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "raw", split="validation[:10%]")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("MahsaVafaie/BZKopen", "raw", split="test[:10%]")
    
