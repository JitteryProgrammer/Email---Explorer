# Email---Explorer

# 📊 Apresentação do meu código
Este é um projeto de classificação de e-mails como spam ou não spam, utilizando as bibliotecas Pandas, scikit-learn, Seaborn e Matplotlib para realizar a análise.

# 🤔 Como esse código futuramente pode ser importante para sua empresa?

Empresas recebem constantemente uma grande quantidade de e-mails, e classificar os e-mails como spam ou não-spam é importante para manter a organização da caixa de entrada e garantir que as mensagens importantes não sejam perdidas.

Além do mais, o o meu  código pode ser facilmente adaptado para classificar outros tipos de dados, como sentimentos em comentários de redes sociais, por exemplo.

# 📈 Atualizações
- Foi aplicado uma Regressão Logística com Grid Search para ajustar a classificação de saída com base em um determinado limiar de probabilidade, gerando uma matriz de confusão e que é avaliada a curva ROC e o score AUC
- Implementei uma função que verifica a validade dos endereços de e-mail carregados a partir de um arquivo CSV e adiciona uma coluna na tabela com o resultado da verificação, em seguida, ele remove todos os endereços inválidos.

# 💻 O que vem a seguir:
- Continuarei a melhorar o desempenho do modelo com técnicas de otimização de hiperparâmetros e feature engineering.
- irei dicionar uma camada de pré-processamento de texto, como a remoção de stopwords e stemming.
- Aumentarei a quantidade de dados disponíveis para treinamento do modelo. 
-   Irei adicionar o Imaplib que  irá permitir  baixar, manipular e enviar e-mails a partir de um servidor IMAP,  ai depois de obter as mensagens de e-mails, elas serão armazenadas em um formato de dados apropriado (como uma lista ou um dataframe do Pandas) e processá-las da mesma forma que o arquivo CSV atual é processado.
