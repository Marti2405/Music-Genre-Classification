import numpy as np

# X_train, Y_train, X_test, Y_test, X_validate, Y_validate

# Chunk list
chunk_list = []

# Import all the chunks
for chunk in chunks:
    chunk_list.append(chunk)


# Concatenate all the chunks
full_data = np.concatenate(chunk_list, axis=0)


# Save the concatenated data as a single numpy file
np.save('full_data.npy', full_data)