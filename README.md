# SMS Spam Classification using-BOW-TF-IDF-Word2Vec-and-AvgWord2Vec

## Overview
This project focuses on classifying SMS messages as either spam or ham (legitimate) using various text vectorization techniques and machine learning models. The dataset used for this task is the **SMS Spam Collection v.1**, which contains 5,574 messages labeled as spam or ham.

## Dataset
The dataset consists of:
- **4,827** ham messages (86.6%)
- **747** spam messages (13.4%)

Each row in the dataset contains two columns:
1. **Label:** 'ham' or 'spam'
2. **Message:** The raw text of the SMS

## Text Vectorization Methods
To convert text data into numerical features, the following techniques are used:

### 1. Bag of Words (BoW)
- Converts text into a fixed-length vector by counting word occurrences.
- Uses `CountVectorizer` from scikit-learn.

### 2. Term Frequency-Inverse Document Frequency (TF-IDF)
- Measures word importance by considering frequency in a document and across all documents.
- Uses `TfidfVectorizer` from scikit-learn.

### 3. Word2Vec
- Captures semantic meaning of words by representing them in a continuous vector space.
- Uses Google's pre-trained Word2Vec model or trains a custom model using Gensim.

### 4. Average Word2Vec (Avg Word2Vec)
- Computes the average word embedding for all words in a message.
- Uses Gensim's Word2Vec embeddings for feature extraction.

## Implementation Steps
1. **Load the dataset**
2. **Preprocess text:**
   - Convert to lowercase
   - Remove punctuation and special characters
   - Tokenization
   - Stopword removal (optional)
   - Lemmatization or stemming (optional)
3. **Feature extraction using BoW, TF-IDF, Word2Vec, and Avg Word2Vec**
4. **Train machine learning models:**
   - Logistic Regression
   - Naive Bayes
   - Support Vector Machine (SVM)
   - Random Forest
   - Neural Networks (optional)
5. **Evaluate models using accuracy, precision, recall, and F1-score**
6. **Compare performance across different vectorization techniques**

## Requirements
Install dependencies using:
```bash
pip install numpy pandas scikit-learn nltk gensim
