#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
from math import log


# In[25]:




# Corpus
sentences = [
    "sunshine state enjoy sunshine",
    "brown fox jump high, brown fox run",
    "sunshine state fox run fast",
]

# Create a vocabulary list by extracting all unique words from the sentences
vocab = set()
for sentence in sentences:
    for word in sentence.split():
        vocab.add(word)
vocab = list(vocab)

# Create a document-term matrix
doc_term_matrix = []
for sentence in sentences:
    # Create a BoW vector for the sentence
    bow_vector = [0] * len(vocab)
    for word in sentence.split():
        if word in vocab:
            bow_vector[vocab.index(word)] += 1
    doc_term_matrix.append(bow_vector)

# Compute the TF vectors
tf_vectors = []
for bow_vector in doc_term_matrix:
    # Normalize the BoW vector by dividing each element by the total number of words in the sentence
    tf_vector = [word_count / len(bow_vector) for word_count in bow_vector]
    tf_vectors.append(tf_vector)

# Compute the IDF values
idf_values = []
for term in vocab:
    # Count the number of documents that contain the term
    doc_count = 0
    for bow_vector in doc_term_matrix:
        if bow_vector[vocab.index(term)] > 0:
            doc_count += 1
    # Compute the IDF value for the term
    idf_value = log(len(sentences) / doc_count)
    idf_values.append(idf_value)

# Calculate the TF.IDF vectors
tfidf_vectors = []
for tf_vector in tf_vectors:
    # Multiply the TF vector by the IDF values
    tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_values)]
    tfidf_vectors.append(tfidf_vector)

# Print the results
print("Vocabulary:", vocab)
print("Document-term matrix:", doc_term_matrix)
print("TF vectors:", tf_vectors)
print("IDF values:", idf_values)
print("TF.IDF vectors:", tfidf_vectors)









# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the strings
S1 = "sunshine state enjoy sunshine"
S3 = "sunshine state fox run fast"

# Create the vectorizer object
vectorizer = TfidfVectorizer()

# Convert the strings to sparse arrays
sparse_array_S1 = vectorizer.fit_transform([S1])
sparse_array_S3 = vectorizer.transform([S3])

# Calculate the cosine similarity
similarity = cosine_similarity(sparse_array_S1, sparse_array_S3)

# Print the similarity
print(similarity)


# In[19]:





# In[20]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




