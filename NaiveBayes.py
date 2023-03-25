# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer as suffix_stripper
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords

# Define constants
DATABASE_PATH = "dataset/Sarcasm_Headlines_Dataset.json"  # path to the dataset
IGNORE_WORD_LIST = "stopwords"  # file containing words to ignore
ROW_IN_SHAPE = 0  # index for row in the dataset shape tuple
DATABASE_COLUMN_HEADLINE_NAME = "headline"  # name of column containing headlines
DATABASE_OUTPUT_COLUMN = 2  # index of column containing output (i.e., whether headline is sarcastic or not)
LANGUAGE = 'english'  # language for stopwords
TEST_SIZE_IN_PER = 0.20  # percentage of data to use for testing
EXCLAMATION_IN_SYMBOL = '!'  # symbol for exclamation points
EXCLAMATION_IN_WORD = ' exclamation'  # word to replace exclamation points with
QUESTION_IN_SYMBOL = '?'  # symbol for question marks
QUESTION_IN_WORD = ' question'  # word to replace question marks with
QUOTATION_IN_SYMBOL = r'\'(.+?)\''  # regex pattern for quoted text
QUOTATION_IN_WORD = ' quotation'  # word to add to headline if it contains quoted text
NOT_LOWERCASE_LETTERS = '[^a-z]'  # regex pattern for non-lowercase letters
SPACE = ' '  # space character
FEATURE_START = 100  # start point for feature vector size
FEATURE_END = 3000  # end point for feature vector size
FEATURE_STEP = 100  # step size for feature vector size
LABEL_NUM_OF_MAX_VEC = 'Number of Max Vector'  # label for x-axis of plot showing max feature vector size
LABEL_ERROR = 'Error'  # label for y-axis of plot showing error rate
LABEL_ACCURACY = 'Accuracy'  # label for y-axis of plot showing accuracy rate
LABEL_PRECISION = 'Precision'  # label for y-axis of plot showing precision rate
LABEL_RECALL = 'Recall'  # label for y-axis of plot showing recall rate

# Load dataset
database = pd.read_json(DATABASE_PATH, lines=True)

# Preprocessing
nltk.download(IGNORE_WORD_LIST)  # download stopwords
stop_words = set(stopwords.words(LANGUAGE))  # create set of stopwords for given language
headlines = []  # list to store processed headlines
output = database.iloc[:, DATABASE_OUTPUT_COLUMN]  # get output column
total_lines = database.shape[ROW_IN_SHAPE]  # get number of rows in dataset

# Loop through all headlines in the dataset
for index in range(total_lines):
    # Replace exclamation points and question marks with appropriate words
    edit = re.sub(EXCLAMATION_IN_SYMBOL, EXCLAMATION_IN_WORD, database[DATABASE_COLUMN_HEADLINE_NAME][index])
    edit = edit.replace(QUESTION_IN_SYMBOL, QUESTION_IN_WORD)
    
    # Check if headline contains quoted text and add appropriate word if it does
    found = re.findall(QUOTATION_IN_SYMBOL, edit)
    if found:
        edit += QUOTATION_IN_WORD
    
    # Replace non-lowercase letters with spaces and split headline into words
    edit = re.sub(NOT_LOWERCASE_LETTERS, SPACE, edit)
    edit = edit.split()
    ss = suffix_stripper()
    edit = [ss.stem(word) for word in edit if word not in stop_words]
    edit = SPACE.join(edit)
    headlines.append(edit)

# Model implementation
features = range(FEATURE_START, FEATURE_END, FEATURE_STEP)
error_rates = []
accuracies = []
precisions = []
recalls = []

for feature in features:
    cv = CountVectorizer(max_features=feature)
    db_input = cv.fit_transform(headlines).toarray()
    train_input, test_input, train_output, test_output = train_test_split(db_input, output, test_size=TEST_SIZE_IN_PER, random_state=0)
    model = GaussianNB()
    model.fit(train_input, train_output)
    test_predictions = model.predict(test_input)
    conf_mat = confusion_matrix(test_output, test_predictions)
    true_positive = conf_mat[0][0]
    false_positive = conf_mat[0][1]
    false_negative = conf_mat[1][0]
    true_negative = conf_mat[1][1]
    total_conf = true_positive + true_negative + false_positive + false_negative
    
    error_rate = (false_positive + false_negative) / total_conf
    accuracy = (true_positive + true_negative) / total_conf
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    error_rates.append(error_rate)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
 #Calculating and printing maximum vectors, error rates, accuracies.   
optimal = features[error_rates.index(min(error_rates))]
print("the optimal numbers of max vectors is %d" % optimal + " with an error of %.2f" % min(error_rates) + " with an accuracy of %.2f" % max(accuracies))

# Plot the performance metrics.
plt.plot(features, error_rates)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_ERROR)
plt.show()

plt.plot(features, accuracies)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_ACCURACY)
plt.show()

plt.plot(features, precisions)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_PRECISION)
plt.show()

plt.plot(features, recalls)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_RECALL)
plt.show()
