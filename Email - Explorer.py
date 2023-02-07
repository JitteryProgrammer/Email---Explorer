import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregando os dados de e-mails
emails = pd.read_csv("> Arquivo CSV<")

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(emails["Exemplo: text"], emails["Exemplo: spam"], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train, y_train)


dump(model, 'model.joblib')


model = load('model.joblib')


predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100))

precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-score: {:.2f}%".format(f1 * 100))

# Plotando a Matriz de Confusão
confusion = confusion_matrix(y_test, predictions)

sns.heatmap(confusion, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


probabilities = model.predict_proba(X_test)



def adjusted_classes(probs, threshold):
    return [1 if p[1] >= threshold else 0 for p in probs]

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
logistic = LogisticRegression()
grid = GridSearchCV(logistic, param_grid, scoring='roc_auc', cv=5)
grid.fit(X_train, y_train)

probabilities = grid.predict_proba(X_test)
pred_classes = adjusted_classes(probabilities, grid.best_params_['threshold'])

threshold = 0.5

confusion = confusion_matrix(y_test, pred_classes)
print("Confusion Matrix with adjusted threshold: \n", confusion)
pred_classes = adjusted_classes(probabilities, threshold)


fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])
roc_auc = roc_auc_score(y_test, probabilities[:, 1])

plt.plot(fpr, tpr, label="AUC = {:.2f}".format(roc_auc))
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()