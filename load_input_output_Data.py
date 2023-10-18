import numpy as np

import os
import numpy as np
from sklearn.model_selection import train_test_split

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

# Load and concatenate input data
X = load_and_concatenate_data(input_data_folder)


# Load and concatenate output data (labels)
Y = load_and_concatenate_data(output_data_folder)

# Split the data into train, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

save_directory = 'Data/Vectors'

# Ensure the directory exists, create it if not
os.makedirs(save_directory, exist_ok=True)

# Save the arrays to .npy files
np.save(os.path.join(save_directory, 'X_train.npy'), X_train)
np.save(os.path.join(save_directory, 'Y_train.npy'), Y_train)
np.save(os.path.join(save_directory, 'X_validate.npy'), X_validate)
np.save(os.path.join(save_directory, 'Y_validate.npy'), Y_validate)
np.save(os.path.join(save_directory, 'X_test.npy'), X_test)
np.save(os.path.join(save_directory, 'Y_test.npy'), Y_test)
