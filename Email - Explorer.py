import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
import email.utils
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from joblib import dump, load

# Leitura do arquivo CSV
emails = pd.read_csv("seu_arquivo.csv")

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(emails["Exemplo: text"], emails["Exemplo: spam"], test_size=0.2, random_state=42)

# Pré-processamento dos dados
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Treinamento e salvamento do modelo
model = MultinomialNB()
model.fit(X_train, y_train)
dump(model, 'model.joblib')

# Carregamento do modelo
model = load('model.joblib')

# Avaliação do modelo usando validação cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-Validation AUC: {cv_scores.mean():.2f}")

# Predições e avaliação do modelo nos dados de teste
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Matriz de Confusão
confusion = confusion_matrix(y_test, predictions)
sns.heatmap(confusion, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Encontrar o melhor threshold usando GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2'], 'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
logistic = LogisticRegression()
pipeline = Pipeline([('smote', SMOTE()), ('logistic', logistic)])
grid = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5)
grid.fit(X_train, y_train)

# Avaliação do modelo com o melhor threshold
probabilities = grid.predict_proba(X_test)
best_threshold = grid.best_params_['logistic__threshold']
pred_classes = [1 if p[1] >= best_threshold else 0 for p in probabilities]

# Matriz de Confusão com o melhor threshold
confusion = confusion_matrix(y_test, pred_classes)
print(f"Confusion Matrix with optimal threshold: \n{confusion}")

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])
roc_auc = roc_auc_score(y_test, probabilities[:, 1])
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# Filtrar e-mails válidos
emails = emails[emails['email_valido']]

# Envio de e-mail com resultados
results = f"Precision: {precision * 100:.2f}%\nRecall: {recall * 100:.2f}%\nF1-score: {f1 * 100:.2f}%"
send_email("email_destinatario@exemplo.com", "Resultados da Classificação de Emails", results)
