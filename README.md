# Music Genre Classifier — Savearth AI Challenge

Modelo de deep learning para classificação automática de géneros musicais a partir de ficheiros de áudio `.wav`, desenvolvido como parte do AI Code Challenge da Savearth.

## Resultado

| Métrica | Valor |
|---|---|
| Accuracy | 88.68% |
| Macro F1-score | 0.89 |

## Como funciona

Cada ficheiro de áudio é dividido em segmentos de 3 segundos. Cada segmento é convertido num **mel-spectrogram** — uma representação visual do som — e passado a uma **CNN** que aprende a distinguir os padrões de cada género musical.

Na previsão, o modelo analisa todos os segmentos da música e combina os resultados (majority voting) para dar uma resposta final com percentagem de confiança.

## Géneros suportados

`blues` `classical` `country` `disco` `hiphop` `jazz` `metal` `pop` `reggae` `rock`

## Estrutura do projecto

## Instalação

```bash
pip install -r requirements.txt
```

## Treinar o modelo

```bash
python train.py
```

## Prever o género de uma música

```bash
python predict.py caminho/para/musica.wav
```

## Abordagem técnica

- Cada música é segmentada em janelas de 3 segundos — passa de ~1000 para ~10000 amostras
- Cada segmento é convertido em mel-spectrogram (128x128 pixels) e normalizado entre 0 e 1
- CNN com 3 camadas convolucionais, BatchNormalization e Dropout para evitar overfitting
- Majority voting na previsão final — combina todos os segmentos para um resultado mais robusto
