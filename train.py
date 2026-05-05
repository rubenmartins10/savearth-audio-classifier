import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

DATASET_PATH = r"dataset (1)\dataset"

# Percorro todas as pastas do dataset, carrego cada ficheiro de áudio
# e extraio os MFCCs como features
def carregar_dados(dataset_path):
    labels = []
    features = []

    for genero in os.listdir(dataset_path):
        pasta_genero = os.path.join(dataset_path, genero)
        if not os.path.isdir(pasta_genero):
            continue

        for ficheiro in os.listdir(pasta_genero):
            if not ficheiro.endswith(".wav"):
                continue

            caminho = os.path.join(pasta_genero, ficheiro)
            try:
                # carrego o áudio (só 30 segundos)
                y, sr = librosa.load(caminho, duration=30)

                # extraio 13 MFCCs e faço a média ao longo do tempo
                # assim cada música fica representada por 13 números
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)

                features.append(mfcc_mean)
                labels.append(genero)

            except Exception as e:
                print(f"✗ Erro em {ficheiro}: {e}")

    return np.array(features), np.array(labels)


print("A carregar os dados...")
X, y = carregar_dados(DATASET_PATH)
print(f"Total: {len(X)} ficheiros | Géneros: {np.unique(y)}\n")

# Converto os géneros (strings) em números para o modelo perceber
# ex: blues=0, classical=1, rock=9, etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Divido em treino (80%) e teste (20%)
# O modelo aprende com o treino e é avaliado no teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Treino o modelo - Random Forest é um bom ponto de partida
# usa várias árvores de decisão e combina os resultados
print("A treinar o modelo...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avalio o modelo no conjunto de teste (dados que nunca viu)
y_pred = modelo.predict(X_test)

# F1-score geral
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"\nF1-score: {f1:.4f}")

# Relatório detalhado por género
print("\nResultados por género:")
print(classification_report(y_test, y_pred, target_names=le.classes_))