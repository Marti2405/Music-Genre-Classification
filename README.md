# Music-Genre-Classification

# Work done

## Data Pre-processing:
A data cleaning script preprocesses the song lyrics and tags.
Processed input and output data is saved in CSV files.
- Input (`processed_input_data.csv`) $\rightarrow$ Contains one row for each song. For each song we have a list of strings (words).
- Ouput  (`processed_output_data.csv`) $\rightarrow$ Contains one row for each song. For each song we have the Genre tag of the song as a string.


## Target (output) data generation
Script that takes the file `processed_output_data.csv` and generates a numpy file `target.npy` wich contains the vectorised tags. 
- pop $\rightarrow$ [1,0,0]
- rap $\rightarrow$ [0,1,0]
- others $\rightarrow$ [0,0,1]


# To do

## X_target (input) data generation
Do a script that takes the file `processed_input_data.csv` and generates a numpy file `x_train.npy` containing vectorised meaning of lyrics.

## Build Neural Network Model
Build a tensorflow NN

## Generate all the data
The processing of data takes a lot of time. Generate all the data.

## Train the model
SPLIT THE DATA (Test, Validation, Train). Train the NN. Do some parameter tunning to prevent overfitting and have better performance.

## Evaluate the model
Do the full model evaluation

## Do poster
Do the poster for the project

## Write report
Write the report for the project.