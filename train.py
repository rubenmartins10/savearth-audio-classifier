import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# O dataset está organizado em pastas - cada pasta é um género musical
# Ex: dataset/blues/ficheiro.wav, dataset/rock/ficheiro.wav
DATASET_PATH = r"dataset (1)\dataset"

# Esta função vai percorrer todas as pastas do dataset
# Para cada ficheiro de áudio, extrai os MFCCs e guarda o género como label
def carregar_dados(dataset_path):
    labels = []    # aqui vou guardar os géneros (blues, rock, etc.)
    features = []  # aqui vou guardar os MFCCs de cada ficheiro

    # percorro cada pasta (cada género)
    for genero in os.listdir(dataset_path):
        pasta_genero = os.path.join(dataset_path, genero)

        # ignoro se não for uma pasta
        if not os.path.isdir(pasta_genero):
            continue

        # percorro cada ficheiro de áudio dentro da pasta do género
        for ficheiro in os.listdir(pasta_genero):
            if not ficheiro.endswith(".wav"):
                continue

            caminho = os.path.join(pasta_genero, ficheiro)

            try:
                # carrego o áudio com librosa (só os primeiros 30 segundos)
                y, sr = librosa.load(caminho, duration=30)

                # extraio os MFCCs - são 13 coeficientes que representam o som
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

                # faço a média de cada coeficiente ao longo do tempo
                # assim fico com um vector fixo de 13 valores por ficheiro
                mfcc_mean = np.mean(mfcc, axis=1)

                # guardo as features e o género correspondente
                features.append(mfcc_mean)
                labels.append(genero)
                print(f"✓ {genero} - {ficheiro}")

            except Exception as e:
                print(f"✗ Erro ao carregar {ficheiro}: {e}")

    return np.array(features), np.array(labels)


# carrego todos os dados
print("A carregar os ficheiros de áudio...")
X, y = carregar_dados(DATASET_PATH)

print(f"\nTotal de ficheiros carregados: {len(X)}")
print(f"Géneros encontrados: {np.unique(y)}")