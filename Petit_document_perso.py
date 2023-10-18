import numpy as np

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