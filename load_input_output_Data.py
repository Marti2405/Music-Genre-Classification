"""
This file loads all the input and output data chunks and saves them in the Data/Vector directory
as a train/test/validation dataset (80/10/10).
"""
import numpy as np
import os
import numpy as np
from sklearn.model_selection import train_test_split


def balance_data(X, Y, n, cats):

    all_indexes = []

    for i, categories in enumerate(cats):
        if categories:
            indexes = np.where(Y[:, i] == 1)[0][:n]
            print(f"for cat: {i} we have {len(indexes)} indexes found")
            all_indexes.extend(indexes)

    Y = Y[all_indexes]
    X = X[all_indexes]

    return X,Y


# Paths to the input and output data folders
# input_data_folder = 'Data/InputDataNotNorm'
input_data_folder = 'Data/InputData'
output_data_folder = 'Data/OutputData'

# Function to load and concatenate data from folder
def load_and_concatenate_data(folder_path):
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            data_list.append(data)
    concatenated_data = np.concatenate(data_list, axis=0)
    return concatenated_data

# Load and concatenate the data
X = load_and_concatenate_data(input_data_folder)
Y = load_and_concatenate_data(output_data_folder)

# Create balanced vectors of shape (size*categories)
#                                 cats= [pop, rap, rock, rb, country, others]
X,Y = balance_data(X, Y, n=50000, cats=[1,1,1,1,1,1])

# Split the data into train, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Ensure the directory exists, create it if not
save_directory = 'Data/VectorsPopRock'
os.makedirs(save_directory, exist_ok=True)

# Save the arrays to .npy files
np.save(os.path.join(save_directory, 'X_train.npy'), X_train)
np.save(os.path.join(save_directory, 'Y_train.npy'), Y_train)
np.save(os.path.join(save_directory, 'X_validate.npy'), X_validate)
np.save(os.path.join(save_directory, 'Y_validate.npy'), Y_validate)
np.save(os.path.join(save_directory, 'X_test.npy'), X_test)
np.save(os.path.join(save_directory, 'Y_test.npy'), Y_test)


print("DONE!")
