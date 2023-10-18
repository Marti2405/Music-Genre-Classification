"""
This file reads the CSV file containing the lyrics data by chunks.
For each row the data is processed.
This file saves the processed data chunks in the folders: Data/InputData & Data/OutputData
This file saves the last chunk processed in a log file.

Content of the saved files:
    Data/InputData -> npy files containing a matrix of the input data.
    Data/OutputData -> npy files containing a matrix of the output data.
"""

import nltk, re, string, logging
import numpy as np
import pandas as pd
import gensim.downloader as api
from nltk.corpus import stopwords



# Load the Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

print("---- Loading of Word2Vec model completed")


# Load the last processed row from the log file if it exists
try:
    with open('data_processing.log', 'r') as log_file:
        lines = log_file.readlines()
        if lines:
            last_processed_row = int(lines[-1].split(':')[-1].strip())  # Extract the last processed row from the log file
        else:
            last_processed_row = 0
except FileNotFoundError:
    last_processed_row = 0


logging.basicConfig(filename='data_processing.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)



def data_clean(text): 
    """
    Cleans the input text by removing metadata, stopwords, and punctuation.
    
    Args:
    text (str): The input text to be cleaned.

    Returns:
    list: A list of cleaned words.
    """
    
    text = re.sub(r'\[.*?\]', '', text) # Remove metadata from text using regular expressions
    text.replace('\n',' ')

    words = nltk.tokenize.wordpunct_tokenize(text)

    text_result = []
    for word in words: # for each word in the tokenized list of words from the sentence
        if word.lower() not in stopwords.words('english') and word[0] not in string.punctuation and word.lower() not in text_result: # if the word not a stop word, word not already in vocab and word not punctuation
            text_result.append(word.lower())
        

    return text_result

    


# Define chunk size (size of part readed and processed from the file)
chunk_size = 1000000
print(f"---- Chunk size = {chunk_size}")


# Define the vectors for "rap", "pop" and others
pop_vector =   np.array(([1, 0, 0]))
rap_vector =   np.array(([0, 1, 0]))
other_vector = np.array(([0, 0, 1])) 

# Define the counter for the number of songs processed
count = last_processed_row
print(f"---- Starting from {count}")

# Iterate through the CSV file in chunks
for chunk in pd.read_csv('./Data/song_lyrics.csv', skiprows=range(1, last_processed_row), chunksize=chunk_size): 
    
    # declare arrays of input and output
    input_data = []
    input_data_non_normalized = []
    output_data = []

    # update counters
    count+= chunk_size

    # for each row (song)
    for index, row in chunk.iterrows():
        
        if index%500==0:
            print(f"---- Iter n°{index}")


        # Process each row
        
        if row['language']=="en": # if the lyrics are in english

            ##_________________Input processing_________________
            sum_word_vecs = np.zeros(300) # initialise sum of word vectors
            total_number_words = 0 # number of words transformed to vectors

            for word_to_vectorize in data_clean(row['lyrics']): # for each word in the cleaned, tokenized list from the lyrics
                try: # vectorise the word
                    sum_word_vecs+= word2vec_model[word_to_vectorize]
                    total_number_words+=1
                except:
                    pass
            tot_vec = sum_word_vecs/total_number_words # mean of all the vectors
            input_data_non_normalized.append(tot_vec)
            input_data.append(tot_vec/np.linalg.norm(tot_vec)) # append the normalized vector to the inpu_data list
            ##____________________________________________________________________

            ##_________________Output processing_________________
            # process output data
            tag = row['tag']
            if tag == 'rap':
                output_data.append(rap_vector)
            elif tag == 'pop':
                output_data.append(pop_vector)
            else:
                output_data.append(other_vector)
            ##____________________________________________________________________

    
    np.save(f'Data/InputData/I_data_chunk_{count}.npy', np.array(input_data)) # save input data matrix
    np.save(f'Data/InputDataNotNorm/I_data_chunk_{count}.npy', np.array(input_data_non_normalized)) # save input data matrix
    np.save(f'Data/OutputData/O_data_chunk_{count}.npy', np.array(output_data)) # save output data matrix
            
    
    

    logging.info(f"Rows processed: {count}") # save info of how many rows have been processed
    print(f"Rows processed: {count}") # print how many rows have been processed
    
    


