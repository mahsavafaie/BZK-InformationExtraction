import os
import json
import gzip
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def predict(model, input_folder, output_file):
    model_name = str(model)

    parent_folder = os.path.dirname(output_file)
    if not os.path.exists(parent_folder) and not parent_folder == '':
        os.makedirs(parent_folder)
    open_func = gzip.open if output_file.endswith('.gz') else open

    already_processed_files = set()
    if os.path.isfile(output_file):
        with open_func(output_file, 'rt', newline='', encoding='utf-8') as jsonfile:
            for line in jsonfile:
                data = json.loads(line)
                already_processed_files.add(data['filename'])

    if len(already_processed_files) > 0:
        logger.info(f"Detected already {len(already_processed_files)} processed files. If you want to start from scratch, delete the output file.")

    with open_func(output_file, 'at', newline='', encoding='utf-8') as jsonfile:        
        # iterate over all files with the .jpg extension in the input folder
        predictions_files = [file for file in os.listdir(input_folder) if file.endswith(".jpg") and file not in already_processed_files]
        logging.info(f"Detected {len(predictions_files)} files that end with .jpg in folder {input_folder} which needs to be processed.")
        image_generator = (Image.open(os.path.join(input_folder, file)) for file in predictions_files)
        for i, results in enumerate(model.predict(image_generator)):
            results['path'] = input_folder
            results['filename'] = predictions_files[i]
            results['model_name'] = model_name
            json.dump(results, jsonfile)
            jsonfile.write('\n')
            jsonfile.flush() # write to disk directly