import numpy as np
import pandas as pd


def get_vectors_info():
    file_path = 'Data/Vectors/'

    # Load the Y_train vector from the file
    Y_total = np.load('Data/OutputData/O_data_chunk_250000.npy')
    Y_train = np.load(file_path + 'Y_train.npy')
    Y_test = np.load(file_path + 'Y_test.npy')
    Y_validate = np.load(file_path + 'Y_validate.npy')

    vectors = [Y_total, Y_train, Y_test, Y_validate]
    vector_names = ['Total', 'Train', 'Test', 'Validate']

    class_labels = ['pop', 'rap', 'others']

    for vector, vector_name in zip(vectors, vector_names):
        print('\n', vector_name)
        # Sum for each class
        class_sums = np.sum(vector, axis=0)

        # Calculate percentages
        total_samples = vector.shape[0]
        class_percentages = class_sums / total_samples * 100

        # Create a dictionary to map labels to percentages
        class_percentage_dict = {label: percentage for label, percentage in zip(class_labels, class_percentages)}

        # Print the class and its percentage

        for label in class_labels:
            print(f'{label}: {class_percentage_dict[label]:.2f}%')



def get_global_counts():

    # Specify the column you want to load
    columns_to_load = ['tag']

    # Load the CSV data with only the 'tag' column
    df = pd.read_csv('Data/song_lyrics.csv', usecols=columns_to_load)

    # Count the tags
    tag_counts = df['tag'].value_counts()

    # Convert the result to a NumPy array
    tag_counts_array = tag_counts.reset_index().values

    # Sort by count in descending order
    sorted_tags = tag_counts_array[np.argsort(tag_counts_array[:, 1])[::-1]]

    # Print the tags and their counts
    for tag, count in sorted_tags:
        print(f"Tag: {tag}, Count: {count}")


get_vectors_info()
# get_global_counts()
