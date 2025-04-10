from typing import Dict, List, Tuple
from datasets.arrow_dataset import Dataset
from PIL.Image import Image
from inferable.models.few_shot.base_few_shot import BaseFewShot
from sentence_transformers import SentenceTransformer, util

class SimilarCPUFewShot(BaseFewShot):

    def __init__(self, model_name: str = 'clip-ViT-B-32', n:int = 1) -> None:
        self.model_name = model_name
        self.model = None
        self.corpus_embeddings = None
        self.data = None
        self.n = n

    def fit(self, training_data: Dataset, validation_dat: Dataset) -> None:
        self.model = SentenceTransformer(self.model_name, device='cpu')

        self.corpus_embeddings = self.model.encode(training_data['image'], convert_to_tensor=True)
        #self.corpus_embeddings = self.corpus_embeddings.to("cuda")
        self.corpus_embeddings = util.normalize_embeddings(self.corpus_embeddings)

        self.data = training_data


    def get_few_shot_examples(self, image : Image) -> List[Tuple[Image, Dict[str, str]]]:
        return self.get_multi_few_shot_examples([image])[0]

    def get_multi_few_shot_examples(self, images : List[Image]) -> List[List[Tuple[Image, Dict[str, str]]]]:
        if not self.data:
            raise ValueError("SimilarFewShot has to be fitted first")
        
        query_embedding = self.model.encode(images, convert_to_tensor=True)
        #query_embedding = query_embedding.to("cuda")
        query_embedding = util.normalize_embeddings(query_embedding)

        all_hits = util.semantic_search(query_embedding, self.corpus_embeddings, score_function=util.dot_score, top_k=self.n)
        
        result_list = []
        for hits in all_hits:
            few_shot_examples = []
            for hit in hits:
                example = self.data[hit['corpus_id']]
                att_value = dict(example)
                example_img = att_value.pop('image')
                few_shot_examples.append((example_img, att_value))
            result_list.append(few_shot_examples)

        return result_list

