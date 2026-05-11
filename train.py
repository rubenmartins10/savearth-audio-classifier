import os
import sys
import argparse
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# aceito o caminho do dataset como argumento de linha de comandos
# assim funciona em Windows, Mac e Linux sem precisar de alterar o código
parser = argparse.ArgumentParser(description='Treinar o modelo CNN para classificação de géneros musicais')
parser.add_argument('--dataset', type=str, default='dataset', help='Caminho para a pasta do dataset')
args = parser.parse_args()

DATASET_PATH = args.dataset
SAMPLE_RATE = 22050
SEGMENT_DURATION = 5   # cada música é dividida em segmentos de 5 segundos
N_MELS = 128
IMG_SIZE = 128


def augmentar_audio(y, sr):
    """
    Aplico variações aleatórias ao áudio bruto antes de converter para mel-spectrogram.
    Isto é data augmentation — cria versões ligeiramente diferentes do mesmo som
    para o modelo aprender a ser robusto a pequenas variações.
    Escolho aleatoriamente uma das técnicas a cada chamada.
    """
    escolha = np.random.randint(0, 4)

    if escolha == 0:
        # pitch shift — mudo o tom da música entre -2 e +2 semitons
        # simula músicas gravadas em diferentes afinações
        n_steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    elif escolha == 1:
        # time stretch — acelero ou abrando ligeiramente a música
        # simula pequenas variações de tempo entre gravações
        rate = np.random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)

    elif escolha == 2:
        # adição de ruído branco — simula ruído de fundo ou microfone imperfeito
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise

    elif escolha == 3:
        # alteração de volume — torno a música mais ou menos alta
        # simula diferentes volumes de gravação
        factor = np.random.uniform(0.7, 1.3)
        y = y * factor

    return y


def audio_para_segmentos(caminho, aumentar=False):
    """
    Carrego o ficheiro de áudio e divido em segmentos de SEGMENT_DURATION segundos.
    Para cada segmento, converto para mel-spectrogram (imagem 2D do som).
    Se aumentar=True, aplico data augmentation ao áudio antes da conversão.
    """
    y, sr = librosa.load(caminho, sr=SAMPLE_RATE)
    segment_length = SEGMENT_DURATION * sr
    segmentos = []

    for start in range(0, len(y) - segment_length, segment_length):
        segment = y[start:start + segment_length]

        # aplico augmentation se pedido (só no treino, nunca no teste)
        if aumentar:
            segment = augmentar_audio(segment, sr)

        # converto o segmento de áudio para mel-spectrogram
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS)

        # converto para decibéis - escala logarítmica mais natural para o ouvido
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # redimensiono para 128x128 pixels - tamanho fixo para a CNN
        mel_resized = tf.image.resize(mel_db[..., np.newaxis], [IMG_SIZE, IMG_SIZE])

        # normalizo entre 0 e 1 para o modelo treinar de forma estável
        mel_norm = (mel_resized - tf.reduce_min(mel_resized)) / (
            tf.reduce_max(mel_resized) - tf.reduce_min(mel_resized) + 1e-6
        )

        segmentos.append(mel_norm.numpy())

    return segmentos


def carregar_dados(dataset_path):
    """
    Percorro todas as pastas do dataset.
    Para cada ficheiro de áudio carrego os segmentos originais E os aumentados.
    Assim duplico o dataset com versões variadas de cada música.
    """
    features = []
    labels = []

    for genero in os.listdir(dataset_path):
        pasta_genero = os.path.join(dataset_path, genero)
        if not os.path.isdir(pasta_genero):
            continue

        print(f"A carregar: {genero}")

        for ficheiro in os.listdir(pasta_genero):
            if not ficheiro.endswith(".wav"):
                continue

            caminho = os.path.join(pasta_genero, ficheiro)
            try:
                # versão original do áudio
                segmentos_originais = audio_para_segmentos(caminho, aumentar=False)
                for seg in segmentos_originais:
                    features.append(seg)
                    labels.append(genero)

                # versão aumentada do mesmo áudio (com pitch shift, ruído, etc.)
                segmentos_aumentados = audio_para_segmentos(caminho, aumentar=True)
                for seg in segmentos_aumentados:
                    features.append(seg)
                    labels.append(genero)

            except Exception as e:
                print(f"  ✗ Erro em {ficheiro}: {e}")

    return np.array(features), np.array(labels)


def criar_modelo_cnn(num_classes):
    """
    Construo a CNN (Convolutional Neural Network).
    Cada camada Conv2D aprende a detectar padrões no mel-spectrogram:
    - camadas iniciais detectam padrões simples (bordas, texturas)
    - camadas mais profundas detectam padrões complexos (ritmo, harmonia)
    BatchNormalization estabiliza o treino.
    Dropout desliga neurónios aleatoriamente para evitar overfitting.
    """
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # bloco 1 - detecta padrões básicos no espectrograma
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # bloco 2 - detecta padrões mais complexos
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # bloco 3 - padrões de alto nível (estrutura musical)
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # bloco 4 - padrões ainda mais abstractos
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # achato o output das convoluções para um vector
        tf.keras.layers.GlobalAveragePooling2D(),

        # camada densa para combinar todas as features aprendidas
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # camada final - uma saída por género
        # softmax converte os valores em probabilidades que somam 1
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return modelo


# --- EXECUÇÃO PRINCIPAL ---

print(f"Dataset: {DATASET_PATH}")
print("A carregar e segmentar os dados (com augmentation)...")
print("Isto vai demorar alguns minutos\n")
X, y = carregar_dados(DATASET_PATH)
print(f"\nTotal de segmentos: {len(X)} | Shape: {X.shape}")

# converto os géneros (strings) em números para o modelo perceber
# blues=0, classical=1, country=2, ... (ordem alfabética)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# divido em treino (80%) e teste (20%)
# stratify garante que todos os géneros ficam equilibrados nos dois conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}\n")

# crio e compilo o modelo
modelo = criar_modelo_cnn(num_classes)
modelo.summary()

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks - acções automáticas durante o treino
callbacks = [
    # para automaticamente se o modelo não melhorar em 15 epochs seguidas
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    # reduz o learning rate se estiver a estagnar
    tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
]

print("\nA treinar a CNN...")
historico = modelo.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks
)

# avaliação final no conjunto de teste (dados que o modelo nunca viu)
print("\nA avaliar...")
y_pred_prob = modelo.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = np.mean(y_pred == y_test)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print("\nResultados por género:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# guardo o modelo para usar no predict.py
modelo.save("modelo_cnn.keras")
print("\nModelo guardado!")

# gero a confusion matrix
# mostra para cada género real, o que o modelo previu
# diagonal perfeita = modelo perfeito
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_,
            yticklabels=le.classes_, cmap='Blues')
plt.title('Confusion Matrix — CNN Music Genre Classifier')
plt.ylabel('Género Real')
plt.xlabel('Género Previsto')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix guardada!")