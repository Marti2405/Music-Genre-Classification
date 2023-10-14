"""
This file reads the CSV file containing the lyrics data.
For each row the data is processed.
This file saves the processed data in the files: processed_input_data.csv & processed_output_data.csv

Content of the saved files:
processed_input_data.csv -> one column with one song per row. List of tokenized and cleaned words from lyrics.
processed_output_data.csv -> one column with one song per row. String refering to the Genre of the song.
"""

import numpy, scipy, math, nltk, re, string, logging
import pandas as pd
import gensim.downloader as api
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('stopwords')
# nltk.download('punkt')



# Load the last processed row from the log file if it exists
try:
    with open('data_processing.log', 'r') as log_file:
        lines = log_file.readlines()
        if lines:
            last_processed_row = int(lines[-1].split(':')[1].strip())  # Extract the last processed row from the log file
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
    stemmer = PorterStemmer()
    text = re.sub(r'\[.*?\]', '', text) # Remove metadata from text using regular expressions
    text.replace('\n','')

    text_result = []
    for word in nltk.tokenize.wordpunct_tokenize(text): # for each word in the tokenized list of words from the sentence

        if word.lower() not in stopwords.words('english') and word[0] not in string.punctuation: # if the word not a stop word, word not already in vocab and word not punctuation
            
            w = stemmer.stem(word.lower())
            if w not in text_result:
                text_result.append(w)
        

    return text_result

    


# Define chunk size (size of part readed and processed from the file)
chunk_size = 10  


count = 0

# Iterate through the CSV file in chunks
for chunk in pd.read_csv('./Data/song_lyrics.csv', skiprows=range(1, last_processed_row), chunksize=chunk_size): 
    

    input_data = []
    output_data = []

    count+= chunk_size

    for index, row in chunk.iterrows():
        # Process each row
        if row['language']=="en": # if the lyrics are in english
            input_data.append(data_clean(row['lyrics']))
            output_data.append(row['tag'])


    # Save input data to the CSV file "processed_input_data.csv" without overwriting the existing file
    with open('processed_input_data.csv', 'a', newline='',encoding='utf-8') as f:
        input_df = pd.DataFrame({'input': input_data})
        input_df.to_csv(f, header=f.tell()==0, index=False)

    # Save output data to the CSV file "processed_output_data.csv" without overwriting the existing file
    with open('processed_output_data.csv', 'a', newline='',encoding='utf-8') as f:
        output_df = pd.DataFrame(output_data, columns=['output'])
        output_df.to_csv(f, header=f.tell()==0, index=False)


    logging.info(f"Rows processed: {count}") # save info of how many rows have been processed
    print(f"Rows processed: {count}") # print how many rows have been processed
    
    


