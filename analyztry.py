import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('bank.csv')
data.drop(['contact', 'day', 'month', 'pdays', 'default', 'previous'], axis=1, inplace=True)

for col in ['job', 'marital', 'education', 'housing', 'loan', 'poutcome']:
    data[col] = data[col].replace('unknown', data[col].mode()[0])
data.head()

data_encoded = pd.get_dummies(data, columns=['job', 'marital', 'education', 'housing', 'loan', 'poutcome'], drop_first=True)

X = data_encoded.drop('deposit', axis=1)
y = data_encoded['deposit']

y = y.map({'yes': 1, 'no': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
