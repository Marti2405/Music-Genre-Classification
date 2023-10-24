"""
This file reads the CSV file ontaining the lyrics data by chunks.
For each row the data is processed.
This file saves the processed data chunks in the folders: Data/InputData & Data/OutputData
This file saves the last chunk processed in a log file.

Content of the saved files:
    Data/InputData -> npy files containing a matrix of the input data.
    Data/InputDataNotNorm -> npy files containing a matrix of the non normalized input data.
    Data/OutputData -> npy files containing a matrix of the output data.
"""

import nltk, re, string, logging
import numpy as np
import pandas as pd
import gensim.downloader as api
from nltk.corpus import stopwords
import time
import threading
import os



START_FROM = int(input("Start from: "))

word2vec_model = api.load("word2vec-google-news-300")
print("---- Loading of Word2Vec model completed")



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




def process_data(start_row=START_FROM, chunk_size=250000, verbose=True):
    empty_songs_counter = 0
    if verbose:
        print(f"---- Chunk size = {chunk_size}")


    # Define the vectors for "rap", "pop" and others
    pop_vector =        np.array([1,0,0,0,0,0])
    rap_vector =        np.array([0,1,0,0,0,0])
    rock_vector =       np.array([0,0,1,0,0,0])
    rb_vector =         np.array([0,0,0,1,0,0])
    country_vector =    np.array([0,0,0,0,1,0])
    other_vector =      np.array([0,0,0,0,0,1])

    # Define the counter for the number of songs processed
    count = start_row
    progress = start_row
    if verbose:
        print(f"---- Starting from {count}")


    timed = time.time()

    # Iterate through the CSV file in chunks
    for chunk in pd.read_csv('./Data/song_lyrics.csv', skiprows=range(1, start_row), chunksize=chunk_size):

        # declare arrays of input and output
        input_data = []
        input_data_non_normalized = []
        output_data = []

        # update counters
        count+= chunk_size

        # for each row (song)
        for index, row in chunk.iterrows():

            # Show progress
            progress+=1
            if progress%500==0:
                if verbose:
                    print(f"Progress -> {progress} || Time taken -> {round(time.time()-timed,2)}")
                    timed = time.time()


            # Process each row

            if row['language']=="en": # if the lyrics are in english
                
                
                ##_________________Input processing_________________
                sum_word_vecs = np.zeros(300) # initialise sum of word vectors
                total_number_words = 0 # number of words transformed to vectors

                for word_to_vectorize in data_clean(row['lyrics']): # for each word in the cleaned, tokenized list from the lyrics
                    # print(word_to_vectorize)
                    try: # vectorise the word
                        sum_word_vecs+= word2vec_model[word_to_vectorize]
                        total_number_words+=1
                    except Exception as e:
                        # print(f"failed on: {word_to_vectorize} error: {e}")
                        pass

                if total_number_words>=10: # if the lyrics contain more than 10 words
                    
                
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
                    elif tag == 'rock':
                        output_data.append(rock_vector)
                    elif tag == 'rb':
                        output_data.append(rb_vector)
                    elif tag == 'country':
                        output_data.append(country_vector)
                    else:
                        output_data.append(other_vector)
                    ##____________________________________________________________________
                else:
                    print(f"Not enough words: {progress}, found only {total_number_words}")
                    empty_songs_counter +=1

        np.save(f'{save_directories[0]}/I_data_chunk_{count}.npy', np.array(input_data)) # save input data matrix
        np.save(f'{save_directories[1]}/I_data_chunk_{count}.npy', np.array(input_data_non_normalized)) # save input data matrix
        np.save(f'{save_directories[2]}/O_data_chunk_{count}.npy', np.array(output_data)) # save output data matrix    
        

        print(f"Number of rows processed: from {start_row} to {count} = {(count-start_row)}") # print how many rows have been processed
        print(f"Number of empty songs encountered: {empty_songs_counter}")


        break # §§§§§§§§§§§§§§§§§§§§§§    REMOVE WHEN REVIEWING  §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§"""

    print("DONE. CHUNK COMPLETED!!!")




# Helper function to calculate the starting indexes for multithreading the whole set
def get_starting_indexes(goal=5000000, chunksize=250000, threads=5):
    starting_indexes = []

    adder = goal/threads
    i=0

    while (threads*i) < goal:
        k = i
        indexes = []
        while k < goal:
            indexes.append(int(k))
            k += adder
        i += chunksize
        starting_indexes.append(indexes)

    return starting_indexes



# ______________________________EXECUTION OF THE FUNCTIONS________________________________________________
# Directories where all the output will be saved
save_directories = ['Data/InputData_Complete', 
                   'Data/InputData_Complete_NotNorm',
                   'Data/OutputData_Complete']
for directory in save_directories:
    os.makedirs(directory, exist_ok=True)


# Call function wich will process only the output and start at the row given as global argument
# process_data(chunk_size=2, verbose=True)



# ________________________________________________________________________________________________________
# SINGLE THREAD, MULTIPLE TERMINALS:
chunksize = 50000
threads = 4

starting_2d = get_starting_indexes(chunksize=chunksize, threads=threads)
starting_2d = np.array(starting_2d)
starting_2d = np.transpose(starting_2d)

for starting_index in starting_2d[START_FROM]:
    process_data(start_row=starting_index, chunk_size=chunksize, verbose=True)
# ________________________________________________________________________________________________________



# ________________________________________________________________________________________________________
# Multi threaded Approach
# chunksize = 50000
# threads = 8

# starting_2d = get_starting_indexes(chunksize=chunksize, threads=threads)

# # for serie in starting_2d:
# def process_data_threaded(start_row):
#     process_data(start_row=start_row, chunk_size=chunksize, verbose=False)

# threads = []

# for starting_index in serie:
#     thread = threading.Thread(target=process_data_threaded, args=(starting_index,))
#     threads.append(thread)
#     thread.start()

# # Wait for all threads to finish
# for thread in threads:
#     thread.join()

# print("\n\n!!!DONE!!! \n\n", serie, "\n\n\n")
# ________________________________________________________________________________________________________
