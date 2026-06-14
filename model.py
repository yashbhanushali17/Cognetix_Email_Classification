#Cognetix Internship Project

#Email Classification Project(intermediate level)

#step 1 import required libraries
import re #regular expressions - for access msgs search , remove , or match patterns inside strings/text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

#step 2 download nltk resources
# nltk.download('stopwords')
# nltk.download('punkt')


#step 3 load dataset 
print("*"*80)
df=pd.read_csv("spam.csv",encoding="latin-1",usecols=[0,1])
df.rename(columns={"v1":"label","v2":"message"},inplace=True)
print(df.head())
print("*"*80)
print("shape of data rows and columns")
print(df.shape)
print("*"*80)
print("data info")
print(df.info())
print("*"*80)
print("missing values if there any : ")
print(df.isnull().sum())
print("*"*80)
print("spam or not spam distribution")
print(df["label"].value_counts())
print("*"*80)

#step 4 text processing
stemmer= PorterStemmer()
stop_words=set(stopwords.words('english'))

def clean_text(text):
    '''
        In this function we gonna process text to clean it.
        
        1 will convert in lower case
        2 will remove non letters , will keep only a-z and space
        3 split long sentence into word (token)
        4 remove stopwords
        5 will apply stemming 
        6 after cleaning again will join a clean sentence
    '''

    #converting in lower
    text=text.lower()

    #removing nonletter

    text=re.sub(r'[^a-z\s]','',text)

    #spliting text in single words
    words=text.split()

    #removing stopwords and apply stemming 

    words=[stemmer.stem(word) for word in words if word not in stop_words]

    #join word to make string using str.join(iterable)

    return ' '.join(words) #this will return  join words while keeping a single space between the words

#calling fucnction in each row of message column

df['cleaned_message']=df['message'].apply(clean_text)

#lets show sample of 5 data
print('cleaned _message\n')
print(df[['message','cleaned_message']].head())
print("*"*80)

#step 5 encoding label column 

df['label_num']=df['label'].map({'ham':0,'spam':1})

print('verfication of encoding : \n')
print(df[['label','label_num']].head())
print(df['label_num'].value_counts())
print("*"*80)

#step 6 TF-IDF vectorization

tfidf=TfidfVectorizer(max_features=5000)

X=tfidf.fit_transform(df['cleaned_message'])
y=df['label_num'].values

#step 7 train-test split

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

#step 8 model training 

#we used multinomialNB because it works well will count based / frequency based data 
model=MultinomialNB()
model.fit(X_train,y_train)

#step 9 model evaluation

y_pred=model.predict(X_test)
accuracy =accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred,target_names=['ham','spam'])
#print evaulation result
print('*'*80)
print(f"Accuracy : {accuracy*100:.4f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1_Score : {f1:.4f}")
print('*'*80)
print("Confusion Matrix : \n")
print(cm)
print('*'*80)
print("Classification Report : \n")
print(cr)


#step - 10 visualization 

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=['Predicted Ham','Predicted Spam'],
    yticklabels=['Actual Ham','Actual spam']
)
plt.title('confusion matrix ')
plt.xlabel("actual label")
plt.ylabel("predicted label")
plt.tight_layout()
plt.savefig('confusion_matrix.png',dpi=150)
plt.show()

#step 11 save model and vectorizer
joblib.dump(model,'spam_classifier_model.pkl')
joblib.dump(tfidf,'tfidf_vectorizer.pkl')

#step 12 predict on new email or sms

def predict_email(text):
    cleaned=clean_text(text)
    vectorized=tfidf.transform([cleaned])
    prediction=model.predict(vectorized)[0]
    if prediction==1:
        return "Spam"
    else:
        return "Not Spam"
    
sample_msg=[
    "Congratulations! You have won a FREE iPhone. Click here to claim your prize now!",

    "Hey, are we still meeting for lunch tomorrow? Let me know!",

    "URGENT: Your bank account has been compromised. Call this number immediately to verify.",

    "Please find the attached report for today's meeting. Let me know if you have any questions.",

    "Win £1000 cash! Reply WIN to this message to enter our FREE prize draw!",

    "Mom, I'll be home by 8pm. Can you please keep dinner ready?",
]

print("\n prediction result \n")
for i , msg in enumerate(sample_msg):
    result = predict_email(msg)
    print(f"Message {i+1} : {msg[:65]}...")
    print(f"Prediction : {result}")
    print("*"*80)