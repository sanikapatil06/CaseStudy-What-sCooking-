import pandas as pd

df = pd.read_json("train.json")
# print(df)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

rfc = RandomForestClassifier()
nb = MultinomialNB()
dtc = DecisionTreeClassifier()
svm = svm.SVC()

dc = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']

# print(df["cuisine"].unique())
x = df['ingredients']
y = df['cuisine'].apply(dc.index)

df['all_ingredients'] = df['ingredients'].map(';'.join)


cv = CountVectorizer()
x = cv.fit_transform(df['all_ingredients'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

dtc.fit(x_train, y_train)
y_dtc = dtc.predict(x_test)

nb.fit(x_train, y_train)
y_nb = nb.predict(x_test)

rfc.fit(x_train, y_train)
y_rfc = rfc.predict(x_test)

svm.fit(x_train, y_train)
y_svm = svm.predict(x_test)


print("Decision Tree:", accuracy_score(y_test, y_dtc))
print("Naive Bayes:", accuracy_score(y_test, y_nb))
print("Random Forest:", accuracy_score(y_test, y_rfc))
print("Support Vector Machine:", accuracy_score(y_test, y_svm))

'''
OUTPUT:
Decision Tree: 0.6368321810182276
Naive Bayes: 0.7323695788812068
Random Forest: 0.7611565053425519
Support Vector Machine: 0.7859208045254557
'''
