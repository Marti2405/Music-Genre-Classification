import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec

# Load the processed input data
input_data = pd.read_csv('processed_input_data.csv')

# Define the parameters for the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Fit the vectorizer on the input data
tfidf_matrix = tfidf_vectorizer.fit_transform(input_data['input'])
print(tfidf_matrix)

# Initialize Doc2Vec and Word2Vec models
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
word2vec_model = Word2Vec(sentences=input_data['input'], vector_size=100, window=5, min_count=1, workers=4)

# Create a list to store the vectors
vectors = []

tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(input_data['input'])]
doc2vec_model.build_vocab(tagged_data)

for index, row in input_data.iterrows():
    
    doc_words = list(row['input'])
    print(type(doc_words))
    print(doc_words)
    doc_vector = doc2vec_model.infer_vector(doc_words)
    tagged_data = [TaggedDocument(words=doc_words, tags=[index])]
    doc2vec_model.build_vocab(tagged_data, update=True)
    print(doc_vector)

exit()
# Iterate over each row and transform the input data into vectors
for index, row in input_data.iterrows():
    doc_words = row['input'].split()
    tagged_data = [TaggedDocument(words=doc_words, tags=[index])]
    doc2vec_model.build_vocab(tagged_data, update=True)
    doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    doc_vector = doc2vec_model.infer_vector(doc_words)
    words = doc_words
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    avg_word_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)
    combined_vector = np.concatenate([doc_vector, avg_word_vector, tfidf_matrix[index].toarray().flatten()])
    vectors.append(combined_vector)

# Convert the list of lists into a numpy array
vectorized_input = np.array(vectors)

print("---- Vectorisation completed")
# Save the numpy array to a file
np.save('input_vectors.npy', vectorized_input)
print("---- Input training data saved")

print(vectorized_input)