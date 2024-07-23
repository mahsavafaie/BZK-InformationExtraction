from inferable.models.base_model import BaseModel
from inferable.data.base_dataset import BaseDataset
import os
import csv
from typing import List
import logging
import time
import datetime

logger = logging.getLogger(__name__)

def evaluate(models : List[BaseModel], datasets : List[BaseDataset], output_folder : str):
    # Evaluate the models on the datasets

    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        training_data, validation_data = dataset.get_training_data(), dataset.get_validation_data()
        test_data = dataset.get_test_data()
        for model in models:
            model_name = str(model)
            # Train the model
            logger.info(f"Run training of model {model_name} on {dataset_name}")
            start_time = time.time()
            model.fit(training_data, validation_data)
            training_time = time.time() - start_time
            training__time_hours_minute_seconds = datetime.timedelta(seconds=training_time)
            logger.info(f"Finished training of model {model_name} on {dataset_name} in {training__time_hours_minute_seconds} (HH:MM:SS).")

            # Predict the metadata
            #logger.info(f"Run prediction of model {model_name} on {dataset_name}")
            #start_time = time.time()
            #predicted_metadata = model.predict(test_data)
            #prediction_time = time.time() - start_time

            # Evaluate the predictions
            #logger.info(f"Run evaluation of model {model_name} on {dataset_name}")
            # Calculate the evaluation metrics
            #with open(os.path.join(output_folder, f"evaluation_results-{model_name}-{dataset_name}.csv"), 'w', newline='', encoding='utf-8') as csvfile:
            #    writer = csv.writer(csvfile)
            #    #writer.writerow(['Image', 'Ground Truth Metadata', 'Predicted Metadata'])
            #    #for i, image in enumerate(test_data):
            #    #    writer.writerow((image, image['metadata'], predicted_metadata[i]))
            #logger.info(f"Finished evaluation of model {model_name} on {dataset_name}")