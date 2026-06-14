# ==================================================
# PROJECT 2 : SPAM EMAIL / SMS CLASSIFIER
# ==================================================
# GOAL    : Build an ML model that classifies messages as Spam or Not Spam (Ham)
# DATASET : SMS Spam Collection Dataset (spam.csv)
# MODEL   : Multinomial Naive Bayes with TF-IDF vectorization
# AUTHOR  : [Your Name]
# ==================================================


# ==================================================
# STEP 1 : IMPORT LIBRARIES
# ==================================================

# --- What is "import"? ---
# "import" tells Python: "go find this library and bring all its tools into our program."
# A library is a collection of pre-written code that someone else built and we can reuse.

import re
# "re" stands for Regular Expressions.
# It is a Python built-in library (no installation needed).
# It gives us tools to search, match, and remove patterns inside strings/text.
# We will use it to remove punctuation, numbers, and special characters from messages.
# Without re, we would have to write very long manual code to clean text.

import numpy as np
# "numpy" is a powerful library for working with numbers and arrays.
# "as np" means: give it a short nickname "np" so we don't have to type "numpy" every time.
# np.array, np.mean etc. are common numpy tools.
# We use numpy here mainly for numerical operations behind the scenes.
# Without numpy, many other libraries (pandas, sklearn) would not even work.

import pandas as pd
# "pandas" is a library for working with tabular data (like Excel sheets in Python).
# "as pd" is a short nickname — standard convention, everyone in the industry uses this.
# pd.DataFrame is pandas's main object — think of it as a Python Excel table.
# pd.read_csv() loads a CSV file into a DataFrame.
# Without pandas, we would have no easy way to load and manage our dataset.

import matplotlib.pyplot as plt
# "matplotlib" is Python's most popular library for drawing graphs and charts.
# "pyplot" is a sub-module inside matplotlib that provides simple plotting functions.
# "as plt" is the standard short nickname used everywhere in the industry.
# We use it to show our confusion matrix heatmap visually.
# Without this, we cannot draw any charts.

import seaborn as sns
# "seaborn" is built on top of matplotlib and makes beautiful, styled charts easily.
# It gives us the heatmap() function to visualize the confusion matrix in a nice way.
# Without seaborn, we would have to write much more code to get the same pretty result.

import joblib
# "joblib" is a library for saving and loading Python objects to disk.
# We use it to save our trained ML model and TF-IDF vectorizer as .pkl files.
# ".pkl" means "pickle file" — a format that stores Python objects.
# Without joblib, every time we run the program we would have to re-train the model,
# which is a waste of time, especially for large datasets.

# --- NLTK Imports ---
import nltk
# "nltk" stands for Natural Language Toolkit.
# It is the most popular Python library for NLP (Natural Language Processing).
# NLP means teaching computers to understand, process, and work with human language.
# Computers don't understand words — they only understand numbers.
# NLTK gives us pre-built tools: stopword lists, stemmers, tokenizers, etc.

from nltk.corpus import stopwords
# "from" means: from inside the "nltk" library, go into the "corpus" section.
# "corpus" in NLP means a large collection of text data.
# "stopwords" is a ready-made list of common English words that carry no real meaning.
# Examples of stopwords: "the", "is", "in", "at", "a", "an", "and", "but"
# These words appear in almost every sentence but don't help us identify spam.
# If we keep them, the model gets confused by noise. So we remove them.

from nltk.stem import PorterStemmer
# "stem" means the root or base form of a word.
# "PorterStemmer" is a popular algorithm that reduces words to their base/root form.
# Example: "running" → "run", "loving" → "love", "called" → "call"
# Why stem? Because "run" and "running" mean the same thing.
# Without stemming, the model treats them as different words, which wastes features.

# --- Scikit-learn Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
# "sklearn" is short for "scikit-learn" — the most popular ML library in Python.
# "feature_extraction.text" is a sub-module for turning text into numbers.
# "TfidfVectorizer" converts a list of text messages into a matrix of numbers.
# TF-IDF stands for Term Frequency - Inverse Document Frequency.
# ML models cannot work with raw text — they need numbers.
# TF-IDF is smarter than simple word counts because it also reduces the importance
# of words that appear too frequently across all documents (like "free", "call").
from sklearn.model_selection import train_test_split
# "train_test_split" splits our dataset into two parts:
#   - Training set : used to teach the model (80% of data)
#   - Testing set  : used to check how well the model learned (20% of data)
# Why split? Because if we test on the same data we trained on,
# the model just "memorizes" answers — this is called overfitting.
# We need unseen data to truly measure model performance.

from sklearn.naive_bayes import MultinomialNB
# "naive_bayes" is a group of simple but powerful ML algorithms based on probability.
# "MultinomialNB" is the variant designed for counting-based data, like word frequencies.
# Naive Bayes is called "Naive" because it assumes all features (words) are independent.
# In reality words are related, but this "naive" assumption still works extremely well for text.
# It is fast, simple, and performs surprisingly well for spam detection.

from sklearn.metrics import (
    accuracy_score,
    # accuracy_score : tells us what percentage of predictions were correct overall.
    precision_score,
    # precision_score : of all messages we predicted as SPAM, how many were actually SPAM?
    # High precision = fewer non-spam messages wrongly sent to spam folder.
    recall_score,
    # recall_score : of all actual SPAM messages, how many did we correctly catch?
    # High recall = fewer spam messages slipping into inbox.
    f1_score,
    # f1_score : the harmonic mean of precision and recall.
    # It gives a single balanced number when both precision and recall matter.
    confusion_matrix,
    # confusion_matrix : a 2x2 table showing correct and incorrect predictions.
    # Rows = actual labels, Columns = predicted labels.
    classification_report
    # classification_report : a full text report showing precision, recall, f1
    # for each class (ham and spam) individually.
)


# ==================================================
# STEP 2 : DOWNLOAD NLTK RESOURCES
# ==================================================

# nltk.download() downloads data files from the internet to your local machine.
# These files are stored locally so we only need to download once.

nltk.download('stopwords')
# Downloads the list of English stopwords.
# Stopwords are common words that appear in almost every sentence but carry no real meaning.
# Examples: "the", "a", "is", "in", "to", "and", "but", "for", "not", "on"
# We remove these because keeping them would just add noise — the model would
# focus on meaningless words instead of the important ones like "free", "win", "prize".

nltk.download('punkt')
# Downloads the tokenizer data.
# A tokenizer splits text into individual words or sentences.
# We may not use it directly but it's a common dependency for NLTK functions.


# ==================================================
# STEP 3 : LOAD DATASET
# ==================================================

print("=" * 55)
print("STEP 3 : LOADING DATASET")
print("=" * 55)

# pd.read_csv() reads a CSV (Comma Separated Values) file from disk.
# It returns a DataFrame — a table with rows and columns, just like an Excel sheet.
# encoding='latin-1' : the spam.csv file uses latin-1 encoding (not the default UTF-8).
# If we use wrong encoding, Python throws a UnicodeDecodeError.
# usecols=[0, 1] : we only load columns at index 0 and 1 (we skip the rest — they are empty).
df = pd.read_csv(
    'spam.csv',               # Name of the CSV file in the same folder as this script
    encoding='latin-1',       # Character encoding used to read the file correctly
    usecols=[0, 1]            # Only load first two columns — v1 (label) and v2 (message)
)

# Rename columns so the names are clear and meaningful.
# "v1" and "v2" are not descriptive — "label" and "message" are much better.
# df.rename() returns a new DataFrame with renamed columns.
# "columns" parameter takes a dictionary: {old_name : new_name}
# "inplace=True" means: don't create a new DataFrame, change this one directly.
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# df.head() shows the first 5 rows of the DataFrame by default.
# It helps us visually verify the data was loaded correctly.
# It's a good habit to always check data after loading.
print("\nFirst 5 rows of the dataset:")
print(df.head())

# df.shape returns a tuple: (number_of_rows, number_of_columns)
# This tells us how big our dataset is.
# E.g.: (5572, 2) means 5572 messages and 2 columns.
print(f"\nDataset Shape (rows, columns): {df.shape}")

# df.info() prints a summary of the DataFrame:
# - Column names
# - Number of non-null (non-missing) values per column
# - Data type of each column (object = text, int64 = integer, etc.)
# "Non-null" means values that are not empty/missing.
print("\nDataset Info:")
df.info()

# df.isnull() returns a DataFrame of True/False — True where a value is missing.
# .sum() adds up the True values per column (True = 1, False = 0).
# This tells us how many missing values exist in each column.
# Missing values can break ML models, so always check.
print("\nMissing Values per Column:")
print(df.isnull().sum())

# df['label'].value_counts() counts how many times each unique label appears.
# This shows us: how many "ham" and how many "spam" messages are in the dataset.
# Knowing the class distribution is important — if one class has much more data,
# the model might be biased towards predicting that class.
print("\nClass Distribution (ham vs spam):")
print(df['label'].value_counts())


# ==================================================
# STEP 4 : TEXT PREPROCESSING
# ==================================================

print("\n" + "=" * 55)
print("STEP 4 : TEXT PREPROCESSING")
print("=" * 55)

# --- Why do we clean text? ---
# Raw text is messy: it has punctuation, numbers, capital letters, stopwords.
# Machines see "Hello!", "hello", "HELLO" as three completely different things.
# Cleaning standardizes text so the model learns patterns, not formatting noise.

# Create a PorterStemmer object.
# This object provides the .stem() method to reduce words to their root form.
# We create it once here so we don't create it inside the function repeatedly.
stemmer = PorterStemmer()

# Load the English stopwords list from NLTK.
# stopwords.words('english') returns a Python list of ~179 common English words.
# We convert it to a set using set() because:
#   - set lookup is O(1) — extremely fast
#   - list lookup is O(n) — slow for large lists
# Since we check thousands of words, speed matters.
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    This function takes a raw text message and returns a cleaned version.

    Steps performed:
    1. Convert to lowercase
    2. Remove all non-letter characters (keep only a-z and spaces)
    3. Split into individual words (tokens)
    4. Remove stopwords
    5. Apply stemming to each word
    6. Join words back into a single string

    Parameter:
        text (str): A single raw message string

    Returns:
        str: A cleaned, preprocessed message string
    """

    # --- Step 4.1 : Convert to Lowercase ---
    # str.lower() converts all uppercase letters to lowercase.
    # "FREE" and "free" are the same word — we want the model to treat them equally.
    # Without this, "Free" and "free" and "FREE" would be seen as 3 different words.
    text = text.lower()

    # --- Step 4.2 : Remove Non-Letter Characters ---
    # re.sub(pattern, replacement, string) finds all matches of "pattern" in "string"
    # and replaces them with "replacement".
    #
    # Pattern explained: '[^a-z\s]'
    #   [ ]     : character class — match any one character inside brackets
    #   ^       : inside [ ], this means "NOT" — so "not these characters"
    #   a-z     : any lowercase letter from a to z
    #   \s      : any whitespace character (space, tab, newline)
    #   [^a-z\s]: match any character that is NOT a letter and NOT a space
    #
    # So this removes: punctuation, numbers, symbols, special characters.
    # We replace them with '' (empty string) — i.e., just delete them.
    # Without this, "win!!!" and "win" would be different tokens.
    text = re.sub(r'[^a-z\s]', '', text)

    # --- Step 4.3 : Split Text into Words (Tokenization) ---
    # str.split() splits a string into a list of words using whitespace as separator.
    # "free money win" → ["free", "money", "win"]
    # This process is called "tokenization" in NLP — breaking text into tokens (words).
    # We need individual words because we process each word separately.
    words = text.split()

    # --- Step 4.4 : Remove Stopwords and Apply Stemming ---
    # This is a list comprehension — a compact way to build a new list in Python.
    # Long version of the same logic:
    #
    #   cleaned_words = []
    #   for word in words:
    #       if word not in stop_words:
    #           stemmed = stemmer.stem(word)
    #           cleaned_words.append(stemmed)
    #
    # "word not in stop_words" : keep only words that are NOT in the stopwords set.
    # "stemmer.stem(word)"     : reduce the word to its root/base form.
    # Example: "winning" → "win", "loves" → "love", "running" → "run"
    # Stemming helps the model see "win", "winning", "winner" as the same concept.
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # --- Step 4.5 : Join Words Back into a String ---
    # str.join(iterable) joins a list of strings into one string.
    # ' '.join(words) puts a single space between each word.
    # ["free", "monei", "win"] → "free monei win"
    # We need a string (not a list) because TF-IDF expects strings as input.
    return ' '.join(words)


# Apply the clean_text() function to every message in the 'message' column.
# df['message'].apply(func) calls func on each value in the column.
# "apply" goes row by row and passes each message into clean_text().
# The result is stored as a new column 'clean_message' in the DataFrame.
# lambda is not used here — we pass our named function directly, which is cleaner.
df['clean_message'] = df['message'].apply(clean_text)

# Show a sample of original vs cleaned messages to verify preprocessing worked.
print("\nSample - Original vs Cleaned Messages:")
print(df[['message', 'clean_message']].head(5))


# ==================================================
# STEP 5 : ENCODE LABELS
# ==================================================

print("\n" + "=" * 55)
print("STEP 5 : ENCODING LABELS")
print("=" * 55)

# --- Why encode labels? ---
# ML models only understand numbers — they cannot work with text like "ham" or "spam".
# We must convert text labels to numbers before feeding data to the model.
# We use a simple mapping: ham = 0, spam = 1
# This is called "label encoding" — encoding categories as numbers.

# df['label'].map() applies a mapping (dictionary) to each value in a column.
# For each value in 'label':
#   if value is 'ham'  → replace with 0
#   if value is 'spam' → replace with 1
# The result is stored as a new column 'label_num' (numeric label).
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Show label distribution after encoding to verify correctness.
print("\nLabel Encoding (ham=0, spam=1):")
print(df['label_num'].value_counts())

# Show a sample to confirm encoding worked correctly.
print("\nSample of label and label_num columns:")
print(df[['label', 'label_num']].head(10))


# ==================================================
# STEP 6 : TF-IDF VECTORIZATION
# ==================================================

print("\n" + "=" * 55)
print("STEP 6 : TF-IDF VECTORIZATION")
print("=" * 55)

# --- Why can't we feed raw text into ML models? ---
# ML models are mathematical functions — they work with numbers, not words.
# A model cannot compute "is 'free money' spam?"
# But it CAN compute "is [0.3, 0.0, 0.8, 0.0, ...] spam?" (a vector of numbers).
# So we convert every message into a vector (array of numbers).

# --- What is TF-IDF? ---
# TF  = Term Frequency  : How often a word appears in ONE specific message.
#       If "win" appears 3 times in a 10-word message → TF = 3/10 = 0.3
# IDF = Inverse Document Frequency : How rare/common a word is across ALL messages.
#       If "win" appears in only 2 out of 5000 messages → IDF is high (rare = important).
#       If "the" appears in all 5000 messages → IDF is very low (common = not important).
# TF-IDF = TF × IDF
# Words that are frequent in a message BUT rare across all messages get high scores.
# Words like "the", "is", "and" get low scores because they appear everywhere.
# This makes TF-IDF smarter than just counting words.

# Create a TfidfVectorizer object.
# max_features=5000 means: use only the top 5000 most important words as features.
# Without this limit, we'd have tens of thousands of features (one per unique word),
# which would make the model slow and might cause overfitting.
# Limiting to 5000 keeps it fast and focused on the most relevant vocabulary.
tfidf = TfidfVectorizer(max_features=5000)

# --- X : Feature Matrix (input to the model) ---
# fit_transform() does two things in one step:
#   1. fit()      : Learns the vocabulary from all clean_message texts
#                   (figures out which 5000 words to use, calculates IDF values)
#   2. transform(): Converts each message into a numerical vector using learned vocab
# The result X is a matrix of shape (num_messages, 5000).
# Each row = one message represented as 5000 numbers.
X = tfidf.fit_transform(df['clean_message'])

# --- y : Label Array (output / target for the model) ---
# These are the correct answers (0=ham, 1=spam) for every message.
# .values converts the pandas Series to a numpy array — sklearn prefers arrays.
y = df['label_num'].values

# Print shape of X to understand our feature matrix dimensions.
# X.shape → (total_messages, 5000)  e.g., (5572, 5000)
print(f"\nTF-IDF Feature Matrix Shape (rows=messages, cols=features): {X.shape}")
print(f"Label Array Shape: {y.shape}")


# ==================================================
# STEP 7 : TRAIN-TEST SPLIT
# ==================================================

print("\n" + "=" * 55)
print("STEP 7 : TRAIN-TEST SPLIT")
print("=" * 55)

# --- Why do we split data? ---
# If we train the model on ALL data and then test on the same data,
# the model just "memorizes" all the correct answers — this is called OVERFITTING.
# Overfitting means the model performs great on training data but fails on new data.
# To truly measure how well the model LEARNED, we must test on data it has NEVER seen.

# train_test_split() randomly splits X and y into training and testing sets.
# The function returns 4 arrays in this order:
#   X_train : features for training (80%)
#   X_test  : features for testing  (20%)
#   y_train : labels for training   (80%)
#   y_test  : labels for testing    (20%)

# test_size=0.2 : reserve 20% of the data for testing, 80% for training.
# This is the industry standard split ratio.

# random_state=42 : sets a fixed "seed" for the random number generator.
# This ensures the split is REPRODUCIBLE — every time you run the code,
# you get the exact same split. Without this, results would differ each run.
# 42 is just a convention — any fixed number works.

X_train, X_test, y_train, y_test = train_test_split(
    X,                # Feature matrix (TF-IDF vectors)
    y,                # Label array (0 or 1)
    test_size=0.2,    # 20% goes to test, 80% goes to train
    random_state=42   # Fixed seed for reproducibility
)

print(f"\nTraining set size : {X_train.shape[0]} messages")
print(f"Testing set size  : {X_test.shape[0]} messages")


# ==================================================
# STEP 8 : TRAIN THE MODEL
# ==================================================

print("\n" + "=" * 55)
print("STEP 8 : TRAINING THE MODEL (Multinomial Naive Bayes)")
print("=" * 55)

# --- What is Naive Bayes? ---
# Naive Bayes is a probability-based ML algorithm built on Bayes' Theorem.
# Bayes' Theorem calculates the probability of something being true
# given some evidence we already know.
#
# For spam detection:
# P(spam | message) = P(message | spam) × P(spam) / P(message)
#
# In simple words: "Given this message contains words like 'free', 'win', 'prize',
# what is the probability it is spam?"
#
# WHY "Naive"?
# It assumes that all words in a message are INDEPENDENT of each other.
# In reality, "free" and "win" are related (they often appear together in spam).
# But even with this "naive" assumption, the algorithm works surprisingly well for text.
#
# WHY MultinomialNB?
# "Multinomial" means it works with count-based / frequency-based data.
# TF-IDF values are frequency-based scores — perfect for MultinomialNB.
# (Other variants like GaussianNB work with continuous data like height/weight.)

# Create the Multinomial Naive Bayes model object.
# This just creates an empty model with default settings — it hasn't learned anything yet.
model = MultinomialNB()

# model.fit() is where the ACTUAL training happens.
# "fit" means: teach the model by showing it data + correct answers.
# X_train : the TF-IDF vectors of training messages (what the model sees as input)
# y_train : the correct labels for those messages (0=ham, 1=spam)
# The model internally calculates:
#   - The probability of each word appearing in spam messages
#   - The probability of each word appearing in ham messages
#   - The overall probability of spam vs ham
# After fit(), the model has "learned" the patterns.
model.fit(X_train, y_train)

print("\nModel training complete!")


# ==================================================
# STEP 9 : MODEL EVALUATION
# ==================================================

print("\n" + "=" * 55)
print("STEP 9 : MODEL EVALUATION")
print("=" * 55)

# --- Why evaluate the model? ---
# Evaluation tells us how well the model performs on data it has NEVER seen before.
# This simulates real-world usage where new, unknown messages come in.
# Without evaluation, we have no idea if the model is actually useful or not.

# model.predict() makes predictions on the test set.
# X_test contains TF-IDF vectors of messages the model has NEVER seen during training.
# The model returns a list of predictions: 0 (ham) or 1 (spam) for each test message.
y_pred = model.predict(X_test)

# --- Understanding the Confusion Matrix ---
# A confusion matrix is a 2x2 table that shows:
#
#                       Predicted: Ham    Predicted: Spam
# Actual: Ham    →   True Negative (TN) | False Positive (FP)
# Actual: Spam   →   False Negative (FN)| True Positive  (TP)
#
# TP (True Positive)  : Spam correctly predicted as Spam  ← GOOD
# TN (True Negative)  : Ham correctly predicted as Ham    ← GOOD
# FP (False Positive) : Ham wrongly predicted as Spam     ← BAD (legitimate email lost!)
# FN (False Negative) : Spam wrongly predicted as Ham     ← BAD (spam slips through!)

# Calculate all evaluation metrics.

# accuracy_score compares y_test (actual) with y_pred (predicted).
# Accuracy = (TP + TN) / Total predictions
# It tells us: overall, what fraction of predictions were correct?
accuracy = accuracy_score(y_test, y_pred)

# Precision = TP / (TP + FP)
# Of all messages we labeled as SPAM, how many were actually SPAM?
# High precision = fewer false alarms (legitimate emails wrongly marked as spam).
precision = precision_score(y_test, y_pred)

# Recall = TP / (TP + FN)
# Of all actual SPAM messages, how many did we catch?
# High recall = fewer spam messages slipping into inbox undetected.
recall = recall_score(y_test, y_pred)

# F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
# This is the harmonic mean of precision and recall.
# It gives a single balanced score — useful when both metrics matter equally.
# If precision is 90% but recall is 10%, F1 helps us see the overall picture.
f1 = f1_score(y_test, y_pred)

# Print all metrics in a readable format.
# f-strings (f"...{variable}...") allow embedding variables directly inside strings.
# :.4f means: show the number with 4 decimal places.
print(f"\nAccuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# --- Confusion Matrix ---
# confusion_matrix(y_test, y_pred) computes the 2x2 confusion matrix.
# y_test = actual labels, y_pred = predicted labels
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# --- Classification Report ---
# classification_report() prints a detailed table for each class.
# target_names=['Ham', 'Spam'] labels the rows clearly.
# It shows precision, recall, f1-score, and support for each class.
# "support" = how many actual samples of that class exist in the test set.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


# ==================================================
# STEP 10 : VISUALIZATION — CONFUSION MATRIX HEATMAP
# ==================================================

print("\n" + "=" * 55)
print("STEP 10 : VISUALIZING CONFUSION MATRIX")
print("=" * 55)

# --- Why visualize? ---
# Raw numbers in a confusion matrix are harder to read quickly.
# A colored heatmap makes it immediately obvious which cells are large or small.
# Good visualization makes your project stand out professionally.

# plt.figure() creates a new blank figure (a drawing canvas).
# figsize=(6, 4) sets the size of the figure in inches: width=6, height=4.
# Without this, matplotlib uses a default size which may be too small or too large.
plt.figure(figsize=(6, 4))

# sns.heatmap() draws a colored heatmap from a 2D array (our confusion matrix).
# Parameters explained:
#   cm          : the confusion matrix data (2x2 numpy array)
#   annot=True  : show the number inside each cell of the heatmap
#   fmt='d'     : format numbers as integers ('d' = digit format)
#                 Without this, numbers show as floats like 965.0 instead of 965
#   cmap='Blues': use a blue color gradient — darker = higher value
#   xticklabels : labels for x-axis (columns) → predicted classes
#   yticklabels : labels for y-axis (rows)    → actual classes
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Ham', 'Predicted Spam'],
    yticklabels=['Actual Ham', 'Actual Spam']
)

# plt.title() adds a title to the top of the chart.
# fontsize=13 makes the title text slightly larger for readability.
plt.title('Confusion Matrix - Spam Classifier', fontsize=13)

# plt.ylabel() adds a label on the y-axis (vertical axis).
plt.ylabel('Actual Label')

# plt.xlabel() adds a label on the x-axis (horizontal axis).
plt.xlabel('Predicted Label')

# plt.tight_layout() automatically adjusts spacing so nothing is cut off.
# Without this, labels or titles can overlap with the chart borders.
plt.tight_layout()

# plt.savefig() saves the figure as an image file to disk.
# 'confusion_matrix.png' : filename to save as
# dpi=150 : dots per inch — higher DPI = sharper, higher quality image
# This is useful for reports, GitHub README, or presentations.
plt.savefig('confusion_matrix.png', dpi=150)
print("\nConfusion matrix heatmap saved as 'confusion_matrix.png'")

# plt.show() displays the chart on screen.
# On some environments (like Jupyter Notebook), it renders inline.
plt.show()


# ==================================================
# STEP 11 : SAVE MODEL AND VECTORIZER
# ==================================================

print("\n" + "=" * 55)
print("STEP 11 : SAVING MODEL AND VECTORIZER")
print("=" * 55)

# --- Why save the model? ---
# Training takes time (and sometimes a lot of computing power).
# Once trained, we don't want to re-train every time we want to make a prediction.
# By saving the model, we can load it later and predict instantly — no retraining needed.
# This is how real-world ML applications work:
#   Phase 1: Train once, save the model.
#   Phase 2: Load saved model, predict on new data.

# joblib.dump(object, filename) serializes a Python object and saves it to disk.
# "serialize" means converting the object into a format that can be stored as a file.
# The file format is .pkl (pickle) — a standard Python serialization format.

# Save the trained Naive Bayes model.
joblib.dump(model, 'spam_model.pkl')
print("Trained model saved as 'spam_model.pkl'")

# Save the TF-IDF vectorizer.
# WHY save the vectorizer too?
# The vectorizer learned a vocabulary from the training data.
# When we get a new message, we must transform it using THE SAME vocabulary.
# If we create a new TfidfVectorizer, it would use a different vocabulary,
# and the model's predictions would be completely wrong.
# So we must always use the same vectorizer that was used during training.
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")


# ==================================================
# STEP 12 : PREDICT NEW EMAIL / SMS
# ==================================================

print("\n" + "=" * 55)
print("STEP 12 : PREDICTING NEW MESSAGES")
print("=" * 55)

def predict_email(text):
    """
    This function takes a raw email or SMS message as input
    and returns whether it is "Spam" or "Not Spam".

    Steps performed inside:
    1. Clean the raw text using our clean_text() function
    2. Transform using the SAME TF-IDF vectorizer used during training
    3. Predict using the trained Naive Bayes model
    4. Return the result as a readable string

    Parameter:
        text (str): A raw email or SMS message

    Returns:
        str: "SPAM" or "NOT SPAM"
    """

    # Step 1 : Clean the input text using our preprocessing function.
    # This applies the same cleaning steps as during training:
    # lowercase → remove special chars → remove stopwords → stemming
    # It's CRITICAL to apply the same preprocessing — otherwise the
    # input format won't match what the model was trained on.
    cleaned = clean_text(text)

    # Step 2 : Transform the cleaned text into a TF-IDF vector.
    # IMPORTANT: We use tfidf.transform() — NOT tfidf.fit_transform().
    # fit_transform() would re-learn vocabulary from this one message — WRONG.
    # transform() uses the vocabulary already learned during training — CORRECT.
    # [cleaned] — we pass a list because TfidfVectorizer expects an iterable of strings.
    # The result is a vector of shape (1, 5000) — one message, 5000 features.
    vectorized = tfidf.transform([cleaned])

    # Step 3 : Predict using the trained model.
    # model.predict() returns an array — e.g., [1] for spam or [0] for ham.
    # [0] extracts the first (and only) prediction from the array.
    prediction = model.predict(vectorized)[0]

    # Step 4 : Convert numeric prediction back to a human-readable label.
    # prediction == 1 → "SPAM"
    # prediction == 0 → "NOT SPAM"
    if prediction == 1:
        return "SPAM"
    else:
        return "NOT SPAM"


# --- Example Predictions ---
# Test the function with sample messages to verify it works correctly.

sample_messages = [
    # Typical spam message — prize/money/urgent language
    "Congratulations! You have won a FREE iPhone. Click here to claim your prize now!",

    # Normal ham message — friendly casual text
    "Hey, are we still meeting for lunch tomorrow? Let me know!",

    # Spam with urgency and financial promise
    "URGENT: Your bank account has been compromised. Call this number immediately to verify.",

    # Typical ham — work-related message
    "Please find the attached report for today's meeting. Let me know if you have any questions.",

    # Spam with free offer keyword
    "Win £1000 cash! Reply WIN to this message to enter our FREE prize draw!",

    # Ham — personal message
    "Mom, I'll be home by 8pm. Can you please keep dinner ready?",
]

print("\n--- Prediction Results ---\n")

# Loop through each sample message and print prediction.
# enumerate(sample_messages) gives both the index (i) and the message (msg).
# i+1 is used so the count starts from 1 instead of 0 (more readable).
for i, msg in enumerate(sample_messages):
    result = predict_email(msg)
    # Print index, a shortened version of the message, and prediction.
    # msg[:60] : show only first 60 characters of the message (to keep output neat).
    # "..." : indicates the message continues beyond what's shown.
    print(f"Message {i+1}: {msg[:65]}...")
    print(f"Prediction : {result}")
    print("-" * 55)

print("\n" + "=" * 55)
print("PROJECT COMPLETE! All steps executed successfully.")
print("=" * 55)
print("\nOutput files generated:")
print("  ✔ spam_model.pkl         — Trained Naive Bayes model")
print("  ✔ tfidf_vectorizer.pkl   — Fitted TF-IDF vectorizer")
print("  ✔ confusion_matrix.png   — Heatmap visualization")
