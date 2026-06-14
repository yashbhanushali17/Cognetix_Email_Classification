
# 📧 Spam Email / SMS Classifier

A machine learning project that classifies SMS or email messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

---

## 📌 Project Overview

Spam messages are a common problem in emails and SMS communication. This project builds a machine learning model that can automatically identify whether a message is spam or legitimate.

The project uses text preprocessing, TF-IDF vectorization, and a Multinomial Naive Bayes classifier for spam detection.

---

## 🎯 Features

* Clean and preprocess text messages
* Remove stopwords and unwanted characters
* Apply stemming using NLTK
* Convert text into numerical features using TF-IDF
* Train a Multinomial Naive Bayes model
* Evaluate model performance
* Predict spam or non-spam messages
* Save trained model and vectorizer for future use

---

## 📂 Dataset

* **Dataset:** SMS Spam Collection Dataset
* **Source:** Kaggle
* **Total Records:** 5,572 messages
* **Classes:**
  * Ham (Not Spam)
  * Spam

Dataset Link:

[https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

## 🤖 Model Used

### Multinomial Naive Bayes

Multinomial Naive Bayes is a popular algorithm for text classification tasks and performs well for spam detection when combined with TF-IDF features.

---

## 📊 Model Performance

Sample results obtained during testing:

* Accuracy: ~98%
* Precision: ~99%
* Recall: ~94%
* F1 Score: ~96%

> Results may vary slightly depending on dataset version and train-test split.

---

## 📁 Project Structure

```text
Cognetix_Email_Classification/
│
├── model.py
├── spam.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── spam_classifier_model.pkl
├── tfidf_vectorizer.pkl
└── confusion_matrix.png
```

---

## ⚙️ Setup & Installation

### 1. Create Virtual Environment

```bash
python -m venv myenv
```

### 2. Activate Virtual Environment

Windows:

```bash
myenv\Scripts\activate
```

Linux / macOS:

```bash
source myenv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the SMS Spam Collection Dataset from Kaggle and place:

```text
spam.csv
```

inside the project folder.

---

## ▶️ How To Run

Run the project using:

```bash
python model.py
```

After running the file:

* Dataset will be loaded and cleaned
* Text messages will be converted into TF-IDF features
* Model will be trained
* Evaluation results will be shown in the terminal
* Confusion matrix image will be generated
* Trained model will be saved as `spam_classifier_model.pkl`
* TF-IDF vectorizer will be saved as `tfidf_vectorizer.pkl`

---

## 🔮 Example Predictions

| Message                                | Prediction |
| -------------------------------------- | ---------- |
| Congratulations! You won a FREE iPhone | Spam       |
| Win £1000 cash now                    | Spam       |
| Hey, are we meeting tomorrow?          | Not Spam   |
| Please find the attached report        | Not Spam   |
