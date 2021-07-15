import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import re

def extract_csv(filename):
    df = pd.read_csv(filename,sep=',',encoding='latin-1',usecols=[0,1])
    df.rename(columns={'v1':'label','v2':'message'},inplace=True)
    return df

messages = extract_csv(filename='spam.csv')
wordnet = WordNetLemmatizer()
corpus = []
for i in range(len(messages)):
    text_only_messages = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    lowercase_only_messages = text_only_messages.lower()
    splitted_words = lowercase_only_messages.split()
    review = [wordnet.lemmatize(word) for word in splitted_words if word not in stopwords.words("english") ]
    reviews = ' '.join(review)
    corpus.append(reviews)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB()
spam_detecion_model = spam_detection_model.fit(X_train,y_train)

y_pred = spam_detecion_model.predict(X_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)

print(cm)
print(acc)

