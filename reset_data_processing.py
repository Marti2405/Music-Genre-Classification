"""
Use this file to reset the data processing
"""
import logging

START_FROM_BEGINNING = True ### IF THIS VARIABLE IS SET TO TRUE THE PROCESSING WILL START FROM THE BEGINNING, 
                             ### ERASING ALL PREVIOUS PROGRESS

# word2vec_model = api.load("word2vec-google-news-300")

if START_FROM_BEGINNING:
    logging.basicConfig(filename='data_processing.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    # Overwrite the CSV files
    with open('processed_input_data.csv', 'w', newline='', encoding='utf-8') as f:
        f.write('input\n')  # Add the header for the 'input' column
    with open('processed_output_data.csv', 'w', newline='', encoding='utf-8') as f:
        f.write('output\n')  # Add the header for the 'output' column

print("Successfull Reset")