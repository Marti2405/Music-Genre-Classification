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



START_FROM = int(input("Start from: "))
last_processed_row = START_FROM


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

    

def process_data(start_row=START_FROM, chunk_size=250000, process_input=True ,process_output=True):
    # Define chunk size (size of part readed and processed from the file)
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
    print(f"---- Starting from {count}")

    word2vec_model = None
    
    if process_input:
        # Load the Word2Vec model only if input has to be processed
        word2vec_model = api.load("word2vec-google-news-300")
        print("---- Loading of Word2Vec model completed")


    timed = time.time()

    # Iterate through the CSV file in chunks
    # for chunk in pd.read_csv('./Data/song_lyrics.csv', skiprows=range(1, last_processed_row), chunksize=chunk_size): 
    for chunk in pd.read_csv('./Data/song_lyrics.csv', skiprows=range(1, last_processed_row), chunksize=chunk_size): 
        
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
                print(f"Progress -> {progress} || Time taken -> {round(time.time()-timed,2)}")
                timed = time.time()


            # Process each row
            
            if row['language']=="en": # if the lyrics are in english
                
                if process_input:
                    ##_________________Input processing_________________
                    sum_word_vecs = np.zeros(300) # initialise sum of word vectors
                    total_number_words = 0 # number of words transformed to vectors

                    for word_to_vectorize in data_clean(row['lyrics']): # for each word in the cleaned, tokenized list from the lyrics
                        try: # vectorise the word
                            sum_word_vecs+= word2vec_model[word_to_vectorize]
                            total_number_words+=1
                        except:
                            pass

                    if total_number_words==0:
                        tot_vec=np.zeros(300)
                    else:
                        tot_vec = sum_word_vecs/total_number_words # mean of all the vectors
                    
                    input_data_non_normalized.append(tot_vec)
                    input_data.append(tot_vec/np.linalg.norm(tot_vec)) # append the normalized vector to the inpu_data list
                    ##____________________________________________________________________

                if process_output:
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

        if process_input:
            np.save(f'Data/InputData/I_data_chunk_{count}.npy', np.array(input_data)) # save input data matrix
            np.save(f'Data/InputDataNotNorm/I_data_chunk_{count}.npy', np.array(input_data_non_normalized)) # save input data matrix
        if process_output:
            # np.save(f'Data/OutputData/O_data_chunk_{count}.npy', np.array(output_data)) # save output data matrix
            np.save(f'Data/OutputDataExtended/O_data_chunk_{count}.npy', np.array(output_data)) # save output data matrix
                
        
        

        print(f"Number of rows processed: {count}") # print how many rows have been processed

        break

    print("DONE. CHUNK COMPLETED!!!")
        
    


# ______________________________EXECUTION OF THE FUNCTIONS________________________________________________


# Call function wich will process only the output and start at row 0
process_data(start_row=0, chunk_size=250000, process_input=False, process_output=True)


# Multi threaded Approach
# starting_indexes = [0, 250000, 1000000, 2000000, 3000000, 4000000]

# def process_data_threaded(start_row):
#     process_data(start_row=start_row, chunk_size=250000, process_input=False, process_output=True)

# threads = []

# for starting_index in starting_indexes:
#     thread = threading.Thread(target=process_data_threaded, args=(starting_index,))
#     threads.append(thread)
#     thread.start()

# # Wait for all threads to finish
# for thread in threads:
#     thread.join()

# print("DONE!")