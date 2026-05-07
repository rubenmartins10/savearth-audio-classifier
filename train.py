import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = r"dataset (1)\dataset"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3   # cada música é dividida em segmentos de 3 segundos
N_MELS = 128
IMG_SIZE = 128

# Em vez de usar 30 segundos de áudio por música,
# divido cada música em segmentos de 3 segundos
# Assim passo de ~1000 amostras para ~10000 - muito melhor para a CNN
def audio_para_segmentos(caminho):
    y, sr = librosa.load(caminho, sr=SAMPLE_RATE)
    segment_length = SEGMENT_DURATION * sr
    segmentos = []

    # percorro o áudio de 3 em 3 segundos
    for start in range(0, len(y) - segment_length, segment_length):
        segment = y[start:start + segment_length]

        # converto o segmento para mel-spectrogram
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # redimensiono para 128x128
        mel_resized = tf.image.resize(mel_db[..., np.newaxis], [IMG_SIZE, IMG_SIZE])

        # normalizo entre 0 e 1
        mel_norm = (mel_resized - tf.reduce_min(mel_resized)) / (
            tf.reduce_max(mel_resized) - tf.reduce_min(mel_resized) + 1e-6
        )
        segmentos.append(mel_norm.numpy())

    return segmentos


def carregar_dados(dataset_path):
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
                segmentos = audio_para_segmentos(caminho)
                for seg in segmentos:
                    features.append(seg)
                    labels.append(genero)
            except Exception as e:
                print(f"  ✗ Erro em {ficheiro}: {e}")

    return np.array(features), np.array(labels)


def criar_modelo_cnn(num_classes):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return modelo


print("A carregar e segmentar os dados...")
X, y = carregar_dados(DATASET_PATH)
print(f"\nTotal de segmentos: {len(X)} | Shape: {X.shape}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# split por índice para evitar que segmentos da mesma música
# apareçam em treino e teste ao mesmo tempo
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}\n")

modelo = criar_modelo_cnn(num_classes)
modelo.summary()

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
]

print("\nA treinar a CNN...")
historico = modelo.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks
)

print("\nA avaliar...")
y_pred_prob = modelo.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = np.mean(y_pred == y_test)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print("\nResultados por género:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

modelo.save("modelo_cnn.keras")
print("\nModelo guardado!")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_,
            yticklabels=le.classes_, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix guardada!")