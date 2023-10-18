import numpy as np

file_path = 'Data/Vectors/Y_train.npy'

# Load the Y_train vector from the file
Y_train = np.load(file_path)

class_sums = np.sum(Y_train, axis=0)

print(class_sums)