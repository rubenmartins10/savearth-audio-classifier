import sys
import numpy as np
import librosa
import tensorflow as tf

# configurações - têm de ser iguais às do treino
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3
N_MELS = 128
IMG_SIZE = 128

# géneros pela mesma ordem que o LabelEncoder usou no treino (ordem alfabética)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def audio_para_segmentos(caminho):
    """Carrego o ficheiro de áudio e divido em segmentos de 3 segundos"""
    y, sr = librosa.load(caminho, sr=SAMPLE_RATE)
    segment_length = SEGMENT_DURATION * sr
    segmentos = []

    for start in range(0, len(y) - segment_length, segment_length):
        segment = y[start:start + segment_length]

        # converto cada segmento para mel-spectrogram
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = tf.image.resize(mel_db[..., np.newaxis], [IMG_SIZE, IMG_SIZE])
        mel_norm = (mel_resized - tf.reduce_min(mel_resized)) / (
            tf.reduce_max(mel_resized) - tf.reduce_min(mel_resized) + 1e-6
        )
        segmentos.append(mel_norm.numpy())

    return np.array(segmentos)


def prever_genero(caminho_ficheiro):
    """Prevejo o género de um ficheiro de áudio"""

    print(f"\nA analisar: {caminho_ficheiro}")

    # carrego o modelo treinado
    modelo = tf.keras.models.load_model("modelo_cnn.keras")

    # preparo os segmentos do ficheiro
    segmentos = audio_para_segmentos(caminho_ficheiro)
    print(f"Segmentos analisados: {len(segmentos)}")

    # o modelo prevê o género de cada segmento de 3 segundos
    predicoes = modelo.predict(segmentos, verbose=0)

    # faço a média das probabilidades de todos os segmentos
    # isto é chamado "majority voting" - o género final é o mais votado
    media_probabilidades = np.mean(predicoes, axis=0)
    genero_previsto = GENRES[np.argmax(media_probabilidades)]
    confianca = np.max(media_probabilidades) * 100

    print(f"\nResultado:")
    print(f"  Género previsto : {genero_previsto.upper()}")
    print(f"  Confiança       : {confianca:.1f}%")

    print(f"\nProbabilidades por género:")
    for genero, prob in sorted(zip(GENRES, media_probabilidades),
                                key=lambda x: x[1], reverse=True):
        barra = "█" * int(prob * 30)
        print(f"  {genero:<12} {prob*100:5.1f}%  {barra}")

    return genero_previsto, confianca


# uso: python predict.py caminho/para/musica.wav
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py <caminho_para_ficheiro.wav>")
        print("Exemplo: python predict.py musica.wav")
        sys.exit(1)

    caminho = sys.argv[1]
    prever_genero(caminho)