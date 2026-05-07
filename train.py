import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

DATASET_PATH = r"dataset (1)\dataset"

# Agora extraio mais features para além dos MFCCs
# Quanto mais informação der ao modelo, melhor ele consegue distinguir os géneros
def extrair_features(caminho):
    y, sr = librosa.load(caminho, duration=30)

    # MFCCs - representam o timbre do som (13 coeficientes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma - representa as notas musicais presentes (dó, ré, mi, etc.)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral contrast - diferença entre picos e vales do espectro
    # ajuda a distinguir música com muita energia (metal) de música suave (classical)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = junto = np.mean(contrast, axis=1)

    # Zero crossing rate - quantas vezes o sinal cruza o zero por segundo
    # sons percussivos (hiphop) têm valores altos, sons suaves têm valores baixos
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # RMS energy - energia média do sinal (volume)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # junto tudo num único vector de features
    features = np.concatenate([
        mfcc_mean,       # 13 valores
        chroma_mean,     # 12 valores
        contrast_mean,   # 7 valores
        [zcr_mean],      # 1 valor
        [rms_mean]       # 1 valor
    ])

    return features


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
                f = extrair_features(caminho)
                features.append(f)
                labels.append(genero)
            except Exception as e:
                print(f"✗ Erro em {ficheiro}: {e}")

    return np.array(features), np.array(labels)


print("A carregar os dados...")
X, y = carregar_dados(DATASET_PATH)
print(f"Total: {len(X)} ficheiros carregados\n")

# Converto os géneros em números
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Divido em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Normalizo as features - importante quando temos features com escalas diferentes
# ex: RMS vai de 0 a 1, MFCCs podem ir de -500 a 500
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treino o modelo com mais árvores para melhor resultado
print("A treinar o modelo...")
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

# Avalio os resultados
y_pred = modelo.predict(X_test)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nF1-score: {f1:.4f}")
print("\nResultados por género:")
print(classification_report(y_test, y_pred, target_names=le.classes_))