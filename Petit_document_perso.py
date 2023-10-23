import numpy as np
import pandas as pd
import os


def get_vectors_info(vector_folder):
    file_path = vector_folder

    # Load all the output vectors
    data_list = []
    folder_path = 'Data/OutputData_V2'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path2 = os.path.join(folder_path, file_name)
            data = np.load(file_path2)
            data_list.append(data)

    Y_total = np.concatenate(data_list, axis=0)
    Y_train = np.load(file_path + 'Y_train.npy')
    Y_test = np.load(file_path + 'Y_test.npy')
    Y_validate = np.load(file_path + 'Y_validate.npy')

    vectors = [Y_total, Y_train, Y_test, Y_validate]
    vector_names = ['Total', 'Train', 'Test', 'Validate']

    class_labels = ['pop', 'rap', 'others']
    class_labels = ['pop', 'rap', 'rock', 'rb', 'country', 'others']


    for vector, vector_name in zip(vectors, vector_names):
        print('\n', vector_name, ": ", vector.shape[0])
        # Sum for each class
        class_sums = np.sum(vector, axis=0)

        # Calculate percentages
        total_samples = vector.shape[0]
        class_percentages = class_sums / total_samples * 100

        # Create a dictionary to map labels to percentages
        class_percentage_dict = {label: percentage for label, percentage in zip(class_labels, class_percentages)}
        class_count_dict = {label: percentage for label, percentage in zip(class_labels, class_sums)}

        # Print the class and its percentage

        for label in class_labels:
            print(f'{label}: {class_percentage_dict[label]:.2f}%, count: {class_count_dict[label]}')



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


def remove_nans_in_vector(folder_path, remove):
    folder_path = folder_path

    file_paths = [folder_path + 'X_train.npy'
                  , folder_path + 'X_test.npy'
                  , folder_path + 'X_validate.npy'
                  , folder_path + 'Y_train.npy'
                  , folder_path + 'Y_test.npy'
                  , folder_path + 'Y_validate.npy']

    for i in range(len(file_paths)):
        vector_file = file_paths[i]

        array = np.load(vector_file)

        nan_indices = np.where(np.isnan(array).any(axis=1))[0]

        if len(nan_indices) > 0:
            print("Indexes of vectors containing NaN values in ", vector_file)

            print("count of the nan indices: ", len(nan_indices))

            for index in nan_indices:
                # print(index)
                if remove:
                    # Replace NaN vector with a ero vector of same shape
                    zeros_vector = np.zeros(array.shape[1])
                    array[index] = zeros_vector

            if remove:
                # Save the modified array
                np.save(vector_file, array)

        else:
            print("No vectors contain NaN values in ", vector_file)




folder_path = "Data/VectorsTest/"
# get_vectors_info(folder_path)
# get_global_counts()
remove_nans_in_vector(folder_path, remove=False) # List the indices without removing
