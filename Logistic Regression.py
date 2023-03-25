#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer as suffix_stripper
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Constants
DATABASE_PATH = "dataset/Sarcasm_Headlines_Dataset.json"
IGNORE_WORD_LIST = "stopwords"
ROW_IN_SHAPE = 0
DATABASE_COLUMN_HEADLINE_NAME = "headline"
DATABASE_OUTPUT_COLUMN = 2
LANGUAGE = 'english'
TEST_SIZE_IN_PER = 0.20
EXCLAMATION_IN_SYMBOL = '!'
EXCLAMATION_IN_WORD = ' exclamation'
QUESTION_IN_SYMBOL = '?'
QUESTION_IN_WORD = ' question'
QUOTATION_IN_SYMBOL = r'\'(.+?)\''
QUOTATION_IN_WORD = ' quotation'
NOT_LOWERCASE_LETTERS = '[^a-z]'
SPACE = ' '
ITERATIONS = 100000
LEARNING_RATE = 0.01
LABEL_ERROR = 'Error'
LABEL_ACCURACY = 'Accuracy'
LABEL_PRECISION = 'Precision'
LABEL_RECALL = 'Recall'

# Import the dataset
database = pd.read_json(DATABASE_PATH, lines=True)

# Data preprocessing
nltk.download(IGNORE_WORD_LIST)
from nltk.corpus import stopwords as sw

headlines = []
output = database.iloc[:, DATABASE_OUTPUT_COLUMN]
total_lines = database.shape[ROW_IN_SHAPE]

# Loop through each headline in the dataset
for index in range(0, total_lines):
    # Replace exclamation mark with the word 'exclamation'
    edit = re.sub(EXCLAMATION_IN_SYMBOL, EXCLAMATION_IN_WORD, database[DATABASE_COLUMN_HEADLINE_NAME][index])
    # Replace question mark with the word 'question'
    edit = edit.replace(QUESTION_IN_SYMBOL, QUESTION_IN_WORD)
    # Find any text in single quotes and append the word 'quotation' to the headline
    found = re.findall(QUOTATION_IN_SYMBOL, edit)
    if found:
        edit += QUOTATION_IN_WORD
    # Remove any characters that are not lowercase letters
    edit = re.sub(NOT_LOWERCASE_LETTERS, SPACE, edit)
    # Split the headline into individual words
    edit = edit.split()
    # Strip any suffixes from each word using the PorterStemmer algorithm and remove stopwords
    ss = suffix_stripper()
    edit = [ss.stem(word) for word in edit if not word in sw.words(LANGUAGE)]
    # Join the words back together into a string
    edit = SPACE.join(edit)
    # Add the processed headline to the list of headlines
    headlines.append(edit)

# Split the dataset into training and testing
train_input, test_input, train_output, test_output = train_test_split(
    headlines, output, test_size=TEST_SIZE_IN_PER, random_state=0)

# Feature extraction
# Convert the text data into a matrix of token counts using CountVectorizer
cv = CountVectorizer(ngram_range=(1, 3))
cv_train_input = cv.fit_transform(train_input)
cv_test_input = cv.transform(test_input)

# Train a logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(cv_train_input, train_output)

# Test the model and calculate performance metrics
test_output_prediction = logistic_regression.predict(cv_test_input)
conf_mat = confusion_matrix(test_output, test_output_prediction)
true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]
total_conf = true_positive + true_negative + false_positive + false_negative
error_rate = (false_positive + false_negative) / total_conf
accuracy = (true_positive + true_negative) / total_conf
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

# Print the performance metrics
print(accuracy)
print(error_rate)
print(precision)
print(recall)
