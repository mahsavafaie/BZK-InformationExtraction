import csv
from collections import OrderedDict, defaultdict

#to compare the average values of norm_edit_dist for each key

def get_sorted_edit_distances(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        
        # Initialize an empty dictionary to hold the relevant values
        edit_distances = {}

        # Iterate over each row in the CSV file
        for row in reader:
            # Find the specific row named 'average_all_images'
            if row.get("image") == "average_all_images":
                # Loop through columns and find those ending with "edit_dist"
                for column_name, value in row.items():
                    if column_name.endswith("norm_edit_dist"):
                        # Convert value to a float and add to dictionary
                        edit_distances[column_name] = float(value)
                break  # Exit after finding the desired row

        # Sort dictionary by values and return as OrderedDict
        sorted_edit_distances = OrderedDict(
            sorted(edit_distances.items(), key=lambda item: item[1])
        )

        print(f"{'Column Name':<30} {'Value':<10}")
        print("-" * 40)
        for key, value in sorted_edit_distances.items():
            print(f"{key:<30} {value:<10}")

        return sorted_edit_distances
    

#to compare the average values of norm_edit_dist for each key, when the values are not empty

def get_sorted_edit_distances_nonempty(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        
        # Initialize an empty dictionary to hold the relevant values
        edit_distances = {}

        # Iterate over each row in the CSV file
        for row in reader:
            # Find the specific row named 'average_all_images'
            if row.get("image") == "average_non_empty_comparisons":
                # Loop through columns and find those ending with "edit_dist"
                for column_name, value in row.items():
                    if column_name.endswith("norm_edit_dist"):
                        # Convert value to a float and add to dictionary
                        edit_distances[column_name] = float(value)
                break  # Exit after finding the desired row

        # Sort dictionary by values and return as OrderedDict
        sorted_edit_distances = OrderedDict(
            sorted(edit_distances.items(), key=lambda item: item[1])
        )

        print(f"{'Column Name':<30} {'Value':<10}")
        print("-" * 40)
        for key, value in sorted_edit_distances.items():
            print(f"{key:<30} {value:<10}")

        return sorted_edit_distances
    

def average_edit_distances_per_layout(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        
        # Dictionary to accumulate totals, counts, and sizes by layout_class
        edit_dist_totals = defaultdict(lambda: {'edit_distance': 0, 'edit_distance_non_empty': 0, 'count': 0})

        # Loop through rows to accumulate values by layout_class
        for row in reader:
            layout_class = row["layout_class"]
            
            # Extract and validate avg_normalized_edit_distance
            edit_distance_str = row["avg_normalized_edit_distance"]
            if edit_distance_str:  # Check if the cell is not empty
                edit_distance = float(edit_distance_str)
                edit_dist_totals[layout_class]['edit_distance'] += edit_distance
            
            # Extract and validate avg_normalized_edit_distance_non_empty
            edit_distance_non_empty_str = row["avg_normalized_edit_distance_non_empty"]
            if edit_distance_non_empty_str:  # Check if the cell is not empty
                edit_distance_non_empty = float(edit_distance_non_empty_str)
                edit_dist_totals[layout_class]['edit_distance_non_empty'] += edit_distance_non_empty

            # Increment count for averaging if at least one of the values is non-empty
            if edit_distance_str or edit_distance_non_empty_str:
                edit_dist_totals[layout_class]['count'] += 1

        # Calculate averages and store them in a new dictionary
        averaged_distances = {
            layout_class: {
                "avg_edit_distance": totals['edit_distance'] / totals['count'] if totals['count'] > 0 else 0,
                "avg_edit_distance_non_empty": totals['edit_distance_non_empty'] / totals['count'] if totals['count'] > 0 else 0,
                "group_size": totals['count']
            }
            for layout_class, totals in edit_dist_totals.items()
        }

        # Sort by "avg_edit_distance" and convert to OrderedDict
        sorted_averages = OrderedDict(
            sorted(averaged_distances.items(), key=lambda item: item[1]["avg_edit_distance"])
        )

        # Print the result as a dictionary-like output with group sizes
        print(f"{'Layout Class':<20} {'Avg Edit Distance':<20} {'Avg Non-Empty Edit Distance':<30} {'Group Size':<10}")
        print("-" * 80)
        for layout_class, averages in sorted_averages.items():
            print(f"{layout_class:<20} {averages['avg_edit_distance']:<20} {averages['avg_edit_distance_non_empty']:<30} {averages['group_size']:<10}")

        return sorted_averages



sorted_edit_distance = average_edit_distances_per_layout("/home/mahi/BZK-InformationExtraction/output/evaluation_results-InternvlModel_InternVL2-40B_1-BZKDatasetRawHF.csv")
print(sorted_edit_distance)