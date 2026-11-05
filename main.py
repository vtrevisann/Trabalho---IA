import lib
from modelos.ANN import NeuralNetwork
from modelos.naiveBayes import NaiveBayes

# I.) Funções para pré-processamento dos datasets:
def carregar_dataset(dataset):
    if dataset == 1:
        dfb = lib.pd.read_csv("Datasets/breast_cancer_winsconsin.csv")
        return dfb
    elif dataset == 2:
        dfm = lib.pd.read_csv("Datasets/train.csv")
        return dfm
    else:
        print("Erro ao carregar o dataset!")

def head(df0): #imprime as primeiras 5 linhas da tabela 
    print(df0.head())

def qtdValClasses(df0):  #quantas entradas existem em cada classe:
    if "diagnosis" in df0.columns:
        print(df0["diagnosis"].value_counts())
    elif "label" in df0.columns:
        print(df0["label"].value_counts())
# ----------------------------------------------------------------------------------------
# II.) Preparação dos dados para o treinamento com modelo:

def separarDados(df0): # Separa atributos (X) e classe (y)
    # Remove colunas desnecessárias, se existirem
    df0 = df0.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    if "diagnosis" in df0.columns:
        X = df0.drop(columns=["diagnosis"])
        y = df0["diagnosis"]
    elif "label" in df0.columns:
        X = df0.drop(columns=["label"])
        y = df0["label"]
    return X, y

def separarTreinoTeste(X,y):
    X_train, X_test, y_train, y_test = lib.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Dados para treinamento
    X_train, X_val, y_train, y_val = lib.train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train) # Dados para teste
    scaler = lib.StandardScaler() # Para padronizar os valores das variáveis antes de treinar o modelo
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_val, y_val 

# ----------------------------------------------------------------------------------------
# III.) Modelos:
def escolherModelo(modeloEscolha):
    X, y = separarDados(df0)
    if modeloEscolha == 1:
         modelo, y_test, y_pred_nb = naiveBayes(X,y)
    elif modeloEscolha == 2:
         modelo, y_test, y_pred_nb = KNN(X,y)
    elif modeloEscolha == 3:
         modelo, y_test, y_pred_nb = ANN(X,y)
    else:
        print("Erro ao escolher o modelo!")
    return modelo, y_test, y_pred_nb

def naiveBayes(X, y):
    X_train, X_test, y_train, y_test, X_val, y_val = separarTreinoTeste(X, y)
    modelo = NaiveBayes()
    modelo.fit(X_train, y_train)
    y_pred_nb = modelo.predict(X_test)
    return modelo, y_test, y_pred_nb

def KNN(X,y):
    return 0
def ANN(X,y):
    X_train, X_test, y_train, y_test, X_val, y_val  = separarTreinoTeste(X,y)
    encoder = lib.LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)
    y_test = encoder.transform(y_test)
    modelo = NeuralNetwork(
    hidden_layer_size=8,
    learning_rate=0.05,
    random_state=42,
    )
    modelo.fit(X_train, y_train, epochs=500)
    y_pred_nb = modelo.predict(X_val)
    return modelo, y_test, y_pred_nb

# ----------------------------------------------------------------------------------------
# IV.) Métricas:
def matrizConfusao(y_test, y_pred, dataset):
    matriz = lib.confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(matriz)
    if dataset == 1:
        lib.plt.figure(figsize=(6, 5))
        lib.sns.heatmap(matriz, annot=True, fmt="d", cmap="Oranges", xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
        lib.plt.xlabel("Classificação prevista")
        lib.plt.ylabel("Classificação real")
        lib.plt.title("Matriz de Confusão (Breast Cancer)")
        lib.plt.show()
    elif dataset == 2: 
        lib.sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=["0","1","2","3","4","5","6","7","8","9"], yticklabels=["0","1","2","3","4","5","6","7","8","9"])
    lib.plt.xlabel("Classificação prevista")
    lib.plt.ylabel("Classificação real")
    lib.plt.title("Matriz de Confusão (Mnist))")
    lib.plt.show()
    return matriz


def metricas(y_test, y_pred):
    acc = lib.accuracy_score(y_test, y_pred)
    prec = lib.precision_score(y_test, y_pred, average="weighted")
    rec = lib.recall_score(y_test, y_pred, average="weighted")
    f1 = lib.f1_score(y_test, y_pred, average="weighted")

    print("Resultados:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    return acc, prec, rec, f1

#def matrizConfusao(X_train, X_test, y_train, y_test):
    #matriz_nb = lib.confusion_matrix(y_test, y_pred_nb, labels=["B", "M"])

    # Printando a matriz sem design:
    #print(lib.matriz_nb)

    #lib.plt.figure(figsize=(6, 5))
    #lib.sns.heatmap(matriz_nb, annot=True, fmt="d", cmap="Oranges", xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
    #lib.plt.xlabel("Classificação prevista")
    #lib.plt.ylabel("Classificação real")
    #lib.plt.title("Matriz de Confusão - Naive Bayes")
    #lib.plt.show()


# ----------------------------------------------------------------------------------------
# V.) Menu Principal

print("""
===========================================================
       Análise Comparativa de Algoritmos de Classificação
===========================================================

Datasets:
[1] Breast Cancer
[2] MNIST

""")

dataset = int(input("Escolha o dataset: "))

print("""
=============================
         MODELOS
=============================

[1] Naive Bayes
[2] KNN
[3] ANN

""")

modeloEscolha = int(input("Escolha o modelo: "))
# ----------------------------------------------------------------------------------------

# VI.) Execução:
df0 = carregar_dataset(dataset)
modelo, y_test, y_pred_nb = escolherModelo(modeloEscolha)

matrizConfusao(y_test, y_pred_nb, dataset)
metricas(y_test, y_pred_nb)