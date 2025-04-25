

from google.colab import files
files.upload()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
# %matplotlib inline

message_dataset = pd.read_csv('emails.csv')
message_dataset.head()

message_dataset.shape

plt.rcParams["figure.figsize" ] = [8,10]
message_dataset.spam.value_counts().plot(kind='pie', autopct='%1.0f%%' )

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english' )



message_dataset['text_without_sw' ] = message_dataset['text'] .apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

message_dataset_spam = message_dataset[message_dataset["spam"] == 1]

plt.rcParams["figure.figsize" ] = [8,10]
text = ' '.join(message_dataset_spam['text_without_sw'])
wordcloud2 = WordCloud().generate(text)

plt.imshow(wordcloud2)
plt.axis("off" )
plt.show()

X = message_dataset["text"]

y = message_dataset["spam"]

def clean_text(doc):
  document = re.sub('[^a-zA-Z]' , ' ' , doc)

  document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

  document = re.sub(r'\s+' , ' ' , document)

  return document

X_sentences = []
reviews = list(X)
for rev in reviews:
  X_sentences.append(clean_text(rev))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7, stop_words=stopwords.words('english' ))
X= vectorizer.fit_transform(X_sentences).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

spam_detector = MultinomialNB()
spam_detector.fit(X_train, y_train)

y_pred = spam_detector.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print (confusion_matrix(y_test,y_pred))
print (classification_report(y_test,y_pred))
print (accuracy_score(y_test,y_pred))

print (X_sentences[56])
print (y[56])

print (spam_detector.predict(vectorizer.transform([X_sentences[56]])))



