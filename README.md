# Music Genre Classifier — Savearth AI Challenge

Modelo de deep learning que ouve um ficheiro de áudio e identifica automaticamente o género musical. Desenvolvido como parte do AI Code Challenge da Savearth.

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 95.82% |
|  Macro F1-score | 0.958 |

## Como funciona

Cada música é dividida em segmentos de 5 segundos. Cada segmento é convertido num mel-spectrogram — uma representação visual do som — e passado a uma CNN treinada para reconhecer padrões em 10 géneros musicais.

Na previsão, o modelo analisa todos os segmentos e combina os resultados para dar uma resposta final com percentagem de confiança.

## Géneros suportados

`blues` `classical` `country` `disco` `hiphop` `jazz` `metal` `pop` `reggae` `rock`

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

## Estrutura do projecto

```
savearth-challenge/
├── train.py            # treino da CNN
├── predict.py          # previsão para ficheiros novos
├── modelo_cnn.keras    # modelo treinado
├── confusion_matrix.png
├── report.md           # relatório técnico com decisões tomadas
├── requirements.txt
└── README.md
```