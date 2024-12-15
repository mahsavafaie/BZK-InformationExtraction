from inferable.models.base_model import BaseModel
from inferable.data.base_dataset import BaseDataset
from inferable.models.utils import get_filename
import pandas as pd
import os
import csv
from typing import List
import logging
import time
import datetime
from nltk import edit_distance
from collections import defaultdict
import xlsxwriter
from PIL import Image
import json


logger = logging.getLogger(__name__)

def format_numbers(number):
    return str(number)#.replace('.', ',') # f'{number:.15f}'.replace('.', ',') # .5

def evaluate(models : List[BaseModel], datasets : List[BaseDataset], output_folder : str, write_images : bool = True):
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        training_data, validation_data = dataset.get_training_data(), dataset.get_validation_data()
        test_data = dataset.get_test_data()

        if write_images:
            for i, image in enumerate(test_data['image']):
                os.makedirs(os.path.join(output_folder, dataset_name), exist_ok=True)
                image.save(os.path.join(output_folder, dataset_name, f"image_{i:03}.jpg"))

        training_data = training_data.remove_columns("Layout class")
        validation_data = validation_data.remove_columns("Layout class")

        keys_to_be_predicted = list(test_data.features.keys())
        keys_to_be_predicted.remove("image")
        keys_to_be_predicted.remove("Layout class")

        for model in models:
            model_name = str(model)
            #process model name such that it can be used a file name
            model_name = model_name.replace("/", "_")

            # Train the model
            logger.info(f"Run training of model {model_name} on {dataset_name}")
            start_time = time.time()
            model.fit(training_data, validation_data)
            training_time = time.time() - start_time
            training_time_hours_minute_seconds = datetime.timedelta(seconds=training_time)
            logger.info(f"Finished training of model {model_name} on {dataset_name} in {training_time_hours_minute_seconds} (HH:MM:SS).")

            # Evaluate the predictions
            logger.info(f"Run evaluation of model {model_name} on {dataset_name}")
            start_time = time.time()
            # Calculate the evaluation metrics
            file_path = str(os.path.join(output_folder, f"evaluation_results-{model_name}-{dataset_name}"))
            with open(file_path + ".csv", 'w', newline='', encoding='utf-8') as csvfile: #,  xlsxwriter.Workbook(file_path + ".xlsx") as xlsxfile:
                writer = csv.writer(csvfile)
                
                header_row = ['image', 'layout_class', 
                    'avg_edit_distance', 'avg_edit_distance_non_empty', 'avg_normalized_edit_distance', 'avg_normalized_edit_distance_non_empty',
                    'hits_t0', 'hits_t0_non_empty', 'hits_t1', 'hits_t1_non_empty', 'hits_t3', 'hits_t3_non_empty']
                for prediction_key in keys_to_be_predicted:
                    header_row.extend([f'{prediction_key}_ground_truth', f'{prediction_key}_predicted', f'{prediction_key}_edit_dist', f'{prediction_key}_norm_edit_dist', 
                                       f'{prediction_key}_t0', f'{prediction_key}_t1', f'{prediction_key}_t3'])
                header_row.extend(['training_time_seconds', 'prediction_time_seconds', 'non_evaluated_attributes'])
                writer.writerow(header_row)
                #xlsxfile.write_row(0, 0,  header_row)

                edit_distance_dict = defaultdict(int)
                normalized_edit_distance_dict = defaultdict(float)
                count_of_non_empty_comparisons = defaultdict(int)
                count_of_empty_comparisons = defaultdict(int)
                all_hits_t0 = defaultdict(int)
                all_hits_t1 = defaultdict(int)
                all_hits_t3 = defaultdict(int)

                prediction_time_sum = 0
                prediction_start_time = time.time()
                for i, predicted in enumerate(model.predict(test_data['image'])):
                    prediction_end_time = time.time()
                    prediction_time = prediction_end_time - prediction_start_time
                    prediction_time_sum += prediction_time

                    ground_truth = test_data[i]
                    file_name = os.path.join(dataset_name, f"image_{i:03}.jpg")
                    row = [file_name, ground_truth['Layout class']]
                    avg_edit_distance = 0
                    avg_normalized_edit_distance = 0
                    non_empty_comparisons_of_row = 0
                    empty_comparisons_of_row = 0
                    hits_t0 = 0
                    hits_t1 = 0
                    hits_t3 = 0

                    for prediction_key in keys_to_be_predicted:
                        ground_truth_value = ground_truth[prediction_key]
                        predicted_value = predicted.get(prediction_key) # assign None if key does not exist

                        # TODO: check if still needed
                        if ground_truth_value is None:
                            ground_truth_value = ""
                        if predicted_value is None:
                            predicted_value = ""

                        computed_edit_distance = edit_distance(predicted_value, ground_truth_value)
                        maximum_length = max(len(predicted_value), len(ground_truth_value))
                        computed_normalized_edit_distance = computed_edit_distance / maximum_length if maximum_length > 0 else 0

                        #print(f"ground_truth_value: {ground_truth_value}")
                        #print(f"predicted_value: {predicted_value}")

                        if ground_truth_value != "" or predicted_value != "": # Only count non-empty comparisons -> if both are empty, we do not count it
                            count_of_non_empty_comparisons[prediction_key] += 1
                            count_of_non_empty_comparisons['all'] += 1
                            non_empty_comparisons_of_row += 1

                            # either non_empty or with empty
                            if computed_edit_distance == 0:
                                hits_t0 += 1
                                all_hits_t0[prediction_key] += 1
                                all_hits_t0['all'] += 1
                            if computed_edit_distance <= 1:
                                hits_t1 += 1
                                all_hits_t1[prediction_key] += 1
                                all_hits_t1['all'] += 1
                            if computed_edit_distance <= 3:
                                hits_t3 += 1
                                all_hits_t3[prediction_key] += 1
                                all_hits_t3['all'] += 1
                        else:
                            count_of_empty_comparisons[prediction_key] += 1
                            count_of_empty_comparisons['all'] += 1
                            empty_comparisons_of_row += 1


                        avg_edit_distance += computed_edit_distance
                        avg_normalized_edit_distance += computed_normalized_edit_distance

                        edit_distance_dict[prediction_key] += computed_edit_distance
                        edit_distance_dict['all'] += computed_edit_distance

                        normalized_edit_distance_dict[prediction_key] += computed_normalized_edit_distance
                        normalized_edit_distance_dict['all'] += computed_normalized_edit_distance

                        row.extend([ground_truth_value, predicted_value, computed_edit_distance, format_numbers(computed_normalized_edit_distance), 
                            '', '', ''
                            #format_numbers(1.0) if computed_edit_distance == 0 else format_numbers(0.0), 
                            #format_numbers(1.0) if computed_edit_distance <= 1 else format_numbers(0.0), 
                            #format_numbers(1.0) if computed_edit_distance <= 3 else format_numbers(0.0)
                        ])
                    
                    # create dict of all not evaluated attributes (predicted dict - keys_to_be_predicted)
                    not_evaluated_keys = set(predicted.keys()) - set(keys_to_be_predicted)
                    not_evaluated_dict = {not_evaluated_key: predicted[not_evaluated_key] for not_evaluated_key in not_evaluated_keys}

                    row.extend(["", prediction_time, json.dumps(not_evaluated_dict)])
                    row.insert(2, format_numbers(avg_edit_distance / len(keys_to_be_predicted)))
                    row.insert(3, format_numbers(avg_edit_distance / non_empty_comparisons_of_row))
                    row.insert(4, format_numbers(avg_normalized_edit_distance / len(keys_to_be_predicted)))                    
                    row.insert(5, format_numbers(avg_normalized_edit_distance / non_empty_comparisons_of_row))
                    row.insert(6, format_numbers((hits_t0 + empty_comparisons_of_row) / len(keys_to_be_predicted)))
                    row.insert(7, format_numbers(hits_t0 / len(keys_to_be_predicted)))
                    row.insert(8, format_numbers((hits_t1 + empty_comparisons_of_row) / len(keys_to_be_predicted)))
                    row.insert(9, format_numbers(hits_t1 / len(keys_to_be_predicted)))
                    row.insert(10, format_numbers((hits_t3 + empty_comparisons_of_row) / len(keys_to_be_predicted)))
                    row.insert(11, format_numbers(hits_t3 / len(keys_to_be_predicted)))

                    writer.writerow(row)
                    #xlsxfile.write_row(i + 1, 0,  row)
                    csvfile.flush() # write to disk directly

                    prediction_start_time = time.time()
                
                test_length = len(test_data)
                all_examples_all_attributes = test_length*len(keys_to_be_predicted)
                avg_row = ['average_all_images', '', # second column is empty because there is no layout class for the average
                           format_numbers(edit_distance_dict['all'] / all_examples_all_attributes), '', format_numbers(normalized_edit_distance_dict['all'] / all_examples_all_attributes), '',
                           format_numbers((all_hits_t0['all'] + count_of_empty_comparisons['all']) / all_examples_all_attributes), '', format_numbers((all_hits_t1['all'] + count_of_empty_comparisons['all']) / all_examples_all_attributes), '', format_numbers((all_hits_t3['all'] + count_of_empty_comparisons['all']) / all_examples_all_attributes), ''
                ]
                for prediction_key in keys_to_be_predicted:
                    avg_row.extend(['', '', 
                                    format_numbers(edit_distance_dict[prediction_key] / test_length), 
                                    format_numbers(normalized_edit_distance_dict[prediction_key] / test_length),
                                    format_numbers((all_hits_t0[prediction_key] + count_of_empty_comparisons[prediction_key]) / test_length),
                                    format_numbers((all_hits_t1[prediction_key] + count_of_empty_comparisons[prediction_key]) / test_length),
                                    format_numbers((all_hits_t3[prediction_key] + count_of_empty_comparisons[prediction_key]) / test_length)
                    ])
                avg_row.extend([training_time, prediction_time_sum])
                writer.writerow(avg_row)
                
                # just in case someone asks why the average of the averages is not the same as the average of all values
                # https://math.stackexchange.com/questions/95909/why-is-an-average-of-an-average-usually-incorrect
                count_of_non_empty_all = count_of_non_empty_comparisons['all'] if count_of_non_empty_comparisons['all'] > 0 else 0
                avg_row_non_empty = ['average_non_empty_comparisons', '',  # second column is empty because there is no layout class for the average
                    '', format_numbers(edit_distance_dict['all'] / count_of_non_empty_all), '', format_numbers(normalized_edit_distance_dict['all'] / count_of_non_empty_all),
                    '', format_numbers(all_hits_t0['all']/ count_of_non_empty_all), '', format_numbers(all_hits_t1['all']/ count_of_non_empty_all), '', format_numbers(all_hits_t3['all']/ count_of_non_empty_all)
                ]
                for prediction_key in keys_to_be_predicted:
                    avg_row_non_empty.extend(['', '', 
                                    format_numbers(edit_distance_dict[prediction_key] / count_of_non_empty_comparisons[prediction_key] if count_of_non_empty_comparisons[prediction_key] > 0 else 0), 
                                    format_numbers(normalized_edit_distance_dict[prediction_key] / count_of_non_empty_comparisons[prediction_key] if count_of_non_empty_comparisons[prediction_key] > 0 else 0),
                                    format_numbers(all_hits_t0[prediction_key] / test_length),
                                    format_numbers(all_hits_t1[prediction_key] / test_length),
                                    format_numbers(all_hits_t3[prediction_key] / test_length)
                    ])
                avg_row_non_empty.extend([training_time, prediction_time_sum])
                writer.writerow(avg_row_non_empty)

                csvfile.flush() # write to disk directly

                
                # TODO: add images to the xlsx file
                # adding images to comments is not supported  https://www.youtube.com/watch?v=pPekR2rzwWI
                # https://github.com/jmcnamara/XlsxWriter/issues/823
                # https://stackoverflow.com/questions/69329336/add-image-into-comment-in-excel
            evaluation_time = time.time() - start_time
            evaluation_time_hours_minute_seconds = datetime.timedelta(seconds=evaluation_time)
            logger.info(f"Finished evaluation of model {model_name} on {dataset_name} in {evaluation_time_hours_minute_seconds} (HH:MM:SS).")


def predict(model, input_folder, output_folder):
    model_name = str(model)
    model_name = model_name.replace("/", "_")

    file_path = str(os.path.join(output_folder, f"prediction_results-{model_name}.jsonl"))
    already_processed_files = set()
    if os.path.exists(output_folder):
        with open(file_path, 'r', newline='', encoding='utf-8') as jsonfile:
            for line in jsonfile:
                data = json.loads(line)
                already_processed_files.add(data['filename'])

    with open(file_path, 'w', newline='', encoding='utf-8') as jsonfile:        
        # iterate ovber all files with the .jpg extension in the input folder
        predictions_files = [file for file in os.listdir(input_folder) if file.endswith(".jpg") and file not in already_processed_files]

        for i, results in enumerate(model.predict([Image.open(os.path.join(input_folder, file)) for file in predictions_files])):
            results['filename'] = predictions_files[i]
            json.dump(results, jsonfile)
            jsonfile.write('\n')
            jsonfile.flush() # write to disk directly