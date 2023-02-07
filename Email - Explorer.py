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
import smtplib
import email.utils

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



def predict_classes(probs, threshold):
    return [1 if p[1] >= threshold else 0 for p in probs]

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
              'penalty': ['l1', 'l2'], 
              'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
logistic = LogisticRegression()
grid = GridSearchCV(logistic, param_grid, scoring='roc_auc', cv=5)

# Aqui é adicionado um passo de balanceamento de dados, como SMOTE
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

# Treinamento com dados balanceados
grid.fit(X_train_resampled, y_train_resampled)

# Previsão com dados de teste
probabilities = grid.predict_proba(X_test)
pred_classes = predict_classes(probabilities, grid.best_params_['threshold'])


threshold = 0.5

confusion = confusion_matrix(y_test, pred_classes)
print("Confusion Matrix with optimal threshold: \n", confusion)

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

def is_valid_email(email):
    try:
        parsed_email = email.utils.parseaddr(email)
        return email.utils.formataddr(parsed_email) == email
    except:
        return False

# Carregando os dados de e-mails
emails = pd.read_csv("> Arquivo CSV<")

# Adiciona uma coluna para verificar se o endereço de e-mail é válido
emails['email_valido'] = emails['Exemplo: email'].apply(is_valid_email)

# Remove todos os endereços de e-mail inválidos
emails = emails[emails['email_valido'] == True]


def send_email(to, subject, message):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("seu_email@gmail.com", "sua_senha")
        message = 'Subject: {}\n\n{}'.format(subject, message)
        server.sendmail("seu_email@gmail.com", to, message)
        print("Email enviado com sucesso para", to)
    except Exception as e:
        print("Ocorreu um erro ao enviar o email:", e)

results = "Precision: {:.2f}%\nRecall: {:.2f}%\nF1-score: {:.2f}%".format(precision * 100, recall * 100, f1 * 100)

send_email("email_destinatario@exemplo.com", "Resultados da Classificação de Emails", results)
