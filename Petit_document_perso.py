import numpy as np
import pandas as pd
import os
import time
import nltk


def get_vectors_info(vector_folder):
    file_path = vector_folder

    # Load all the output vectors
    data_list = []
    folder_path = 'Data/OutputData'
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
    # ________________________________________________________________________________________________
    # COUNT THE NUMBER OF NO LYRICS SONGS PER TAG IN ITERATIONS

    # # Specify the chunk size.
    # chunk_size = 100000

    # # Initialize an empty list to store the results DataFrames.
    # result_dfs = []

    # # Iterate through the CSV file in chunks.
    # for chunk in pd.read_csv('Data/song_lyrics.csv', usecols=['lyrics', 'tag'], chunksize=chunk_size):
    #     # Group by 'tag' and count the empty 'lyrics' in the current chunk.
    #     empty_lyrics_counts = chunk.groupby('tag')['lyrics'].apply(lambda x: (x == '').sum()).reset_index()
        
    #     # Append the result DataFrame for the current chunk to the list.
    #     result_dfs.append(empty_lyrics_counts)
        
    #     # Print the intermediate results for the current chunk.
    #     print("Intermediate Result for Chunk:")
    #     print(pd.concat(result_dfs))
    #     print('\n')

    # # Concatenate all the result DataFrames.
    # result_df = pd.concat(result_dfs, ignore_index=True)

    # # Group the results by 'tag' to get the final count.
    # final_result = result_df.groupby('tag')['lyrics'].sum().reset_index()

    # # Print the final result.
    # print("Final Result:")
    # print(final_result)
    # ________________________________________________________________________________________________



    # ________________________________________________________________________________________________
    # COUNT THE NUMBER OF NO LYRICS SONGS
    # # Load the CSV data with only the 'tag' column
    # df = pd.read_csv('Data/song_lyrics.csv', usecols=['lyrics', 'tag'])

    # start_time = time.time()


    # # Group by 'tag' and count the empty 'lyrics' using the sum of boolean masks.
    # empty_lyrics_counts = df.groupby('tag')['lyrics'].apply(lambda x: (x == '').sum()).reset_index()

    # # Rename the columns for clarity.
    # empty_lyrics_counts.columns = ['tag', 'empty_lyrics_count']

    # # Now, empty_lyrics_counts contains the count of songs with empty 'lyrics' for each 'tag'.
    # print(empty_lyrics_counts)

    # end_time = time.time()
    # # Calculate and print the total time taken.
    # total_time = end_time - start_time
    # print("Total Time Taken:", total_time, "seconds")
    # ________________________________________________________________________________________________




    # ________________________________________________________________________________________________
    # Read the CSV file
    df = pd.read_csv('Data/song_lyrics.csv', usecols=['lyrics', 'tag'])

    # Add a new column with the length of 'lyrics'
    df['lyrics_length'] = df['lyrics'].apply(len)

    # Sort the DataFrame by the 'lyrics_length' column in ascending order
    df = df.sort_values(by='lyrics_length')

    # Drop the 'lyrics_length' column if you don't need it anymore
    df = df.drop(columns='lyrics_length')

    # Convert the DataFrame to a NumPy array
    data_array = df.to_numpy()

    # Save the NumPy array to an .npy file
    np.save('sorted_lyrics.npy', data_array)
    # ________________________________________________________________________________________________




    # ________________________________________________________________________________________________
    # COUNT THE NUMBER OF SONGS PER TAG
    # # Specify the column you want to load
    # columns_to_load = ['tag']

    # # Load the CSV data with only the 'tag' column
    # df = pd.read_csv('Data/song_lyrics.csv', usecols=columns_to_load)

    # # Count the tags
    # tag_counts = df['tag'].value_counts()

    # # Convert the result to a NumPy array
    # tag_counts_array = tag_counts.reset_index().values

    # # Sort by count in descending order
    # sorted_tags = tag_counts_array[np.argsort(tag_counts_array[:, 1])[::-1]]

    # # Print the tags and their counts
    # for tag, count in sorted_tags:
    #     print(f"Tag: {tag}, Count: {count}")
    # ________________________________________________________________________________________________


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



words = nltk.tokenize.wordpunct_tokenize("")



folder_path = "Data/VectorsBalanced_PopRapCountry/"
# get_vectors_info(folder_path)
# get_global_counts() #Broken!
# remove_nans_in_vector(folder_path, remove=True) # List the indices without removing
