# Relatório Técnico — Music Genre Classifier

## Problema

Classificação automática de géneros musicais a partir de ficheiros de áudio `.wav` em 10 categorias: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae e rock.

## Dataset

- **GTZAN Music Genre Dataset** — 1000 ficheiros `.wav` de 30 segundos cada
- 10 géneros com 100 ficheiros por género
- 1 ficheiro corrompido ignorado (`jazz.00054.wav`)

## Abordagem

### 1. Pré-processamento

Em vez de usar os 30 segundos de cada música como uma única amostra, dividi cada ficheiro em **segmentos de 5 segundos**. Esta decisão foi fundamental — passou de ~1000 para ~6000 amostras, o que é muito mais adequado para treinar uma CNN.

Cada segmento é convertido num **mel-spectrogram** de 128x128 pixels. O mel-spectrogram representa o som como uma imagem 2D onde:
- O eixo X representa o tempo
- O eixo Y representa as frequências (escala Mel, logarítmica)
- A intensidade da cor representa a energia em cada frequência

Escolhi mel-spectrogram em vez de MFCC porque preserva mais informação espectral, o que é vantajoso para uma CNN que aprende padrões visuais.

### 2. Data Augmentation

Para aumentar a robustez do modelo, apliquei data augmentation no áudio bruto antes da conversão para mel-spectrogram. Para cada ficheiro original gerei uma versão aumentada com uma das seguintes técnicas escolhida aleatoriamente:

- **Pitch shift** (±2 semitons) — simula músicas em diferentes afinações
- **Time stretch** (0.85x a 1.15x) — simula variações de tempo
- **Ruído branco** — simula gravações com ruído de fundo
- **Alteração de volume** (0.7x a 1.3x) — simula diferentes volumes de gravação

Isto duplicou o dataset para ~12000 amostras de treino.

### 3. Arquitectura CNN

Input (128x128x1)
→ Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
→ Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
→ Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
→ Conv2D(256) + BatchNorm + MaxPool + Dropout(0.30)
→ GlobalAveragePooling2D
→ Dense(256) + Dropout(0.50)
→ Dense(10, softmax)

**Decisões de arquitectura:**
- 4 blocos convolucionais com filtros a duplicar (32→64→128→256) para aprender padrões de complexidade crescente
- BatchNormalization após cada camada para estabilizar o treino
- Dropout progressivo para evitar overfitting
- GlobalAveragePooling2D em vez de Flatten para reduzir parâmetros e melhorar generalização

### 4. Treino

- **Optimizador:** Adam com learning rate 0.001
- **Loss:** Sparse Categorical Crossentropy
- **Batch size:** 64
- **Early stopping:** patience=15 (parou ao epoch 54)
- **ReduceLROnPlateau:** reduz o learning rate quando estagna

### 5. Previsão

Na previsão de um ficheiro novo, o modelo analisa cada segmento de 5 segundos individualmente e combina as probabilidades de todos os segmentos (majority voting). O género final é o que obteve maior probabilidade média — este método torna o modelo mais robusto a segmentos ambíguos.

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 95.82% |
| Macro F1-score | 0.958 |

### Resultados por género

| Género | Precision | Recall | F1-score |
|---|---|---|---|
| blues | 0.98 | 0.98 | 0.98 |
| classical | 0.95 | 0.99 | 0.97 |
| country | 0.97 | 0.89 | 0.93 |
| disco | 0.90 | 0.98 | 0.94 |
| hiphop | 0.98 | 0.98 | 0.98 |
| jazz | 1.00 | 0.96 | 0.98 |
| metal | 0.97 | 0.94 | 0.96 |
| pop | 0.93 | 1.00 | 0.96 |
| reggae | 0.96 | 0.97 | 0.96 |
| rock | 0.95 | 0.89 | 0.92 |

O género com pior desempenho foi **rock** (F1=0.92), provavelmente por partilhar características sonoras com blues e country.

## Conclusão

A combinação de segmentação em janelas de 5 segundos, data augmentation no áudio bruto, e uma CNN com regularização adequada permitiu atingir **95.82% de accuracy** — um resultado acima da média para o dataset GTZAN com CNNs standard.