import os
import json
import gzip
from PIL import Image

def predict(model, input_folder, output_file):
    model_name = str(model)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    open_func = gzip.open if output_file.endswith('.gz') else open

    already_processed_files = set()
    if os.path.isfile(output_file):
        with open_func(output_file, 'rt', newline='', encoding='utf-8') as jsonfile:
            for line in jsonfile:
                data = json.loads(line)
                already_processed_files.add(data['filename'])

    with open_func(output_file, 'at', newline='', encoding='utf-8') as jsonfile:        
        # iterate ovber all files with the .jpg extension in the input folder
        predictions_files = [file for file in os.listdir(input_folder) if file.endswith(".jpg") and file not in already_processed_files]

        for i, results in enumerate(model.predict([Image.open(os.path.join(input_folder, file)) for file in predictions_files])):
            results['path'] = input_folder
            results['filename'] = predictions_files[i]
            results['model_name'] = model_name
            json.dump(results, jsonfile)
            jsonfile.write('\n')
            jsonfile.flush() # write to disk directly