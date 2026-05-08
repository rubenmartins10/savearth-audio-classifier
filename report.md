# Relatório Técnico — Music Genre Classifier

## O Problema

O objetivo era construir um modelo capaz de ouvir um ficheiro de áudio e identificar automaticamente o género musical — blues, jazz, rock, entre outros. É um problema de classificação multi-classe com 10 categorias.

## Dataset

Trabalhei com o dataset GTZAN, que contém 1000 ficheiros de áudio em formato `.wav`, com 30 segundos cada, divididos em 10 géneros com 100 ficheiros por género. Um ficheiro estava corrompido (`jazz.00054.wav`) e foi ignorado automaticamente.

## As decisões que tomei e porquê

### Segmentar em vez de usar o áudio inteiro

A primeira decisão importante foi não usar os 30 segundos de cada música como uma única amostra. Em vez disso, dividi cada ficheiro em segmentos de 5 segundos.

O motivo é simples: uma CNN precisa de dados para aprender. Com 1000 ficheiros teria muito poucas amostras. Ao segmentar em janelas de 5 segundos, passei para ~6000 amostras — muito mais adequado para treinar uma rede neuronal.

### Mel-spectrogram como representação do som

Para o modelo conseguir "ver" o áudio, converti cada segmento num mel-spectrogram — uma imagem 2D onde o eixo X representa o tempo, o eixo Y representa as frequências, e a cor representa a energia em cada frequência.

Escolhi mel-spectrogram em vez de MFCC porque preserva mais informação espectral. Para uma CNN que aprende padrões visuais, mais detalhe é melhor.

### Data augmentation no áudio bruto

Com ~6000 amostras o modelo ainda podia fazer overfitting. Para resolver isso, apliquei data augmentation — para cada ficheiro original gerei uma versão modificada com uma destas técnicas:

- **Pitch shift** — mudo o tom ligeiramente (±2 semitons)
- **Time stretch** — acelero ou abrando o áudio (0.85x a 1.15x)
- **Ruído branco** — adiciono ruído de fundo suave
- **Alteração de volume** — torno o áudio mais alto ou mais baixo

Isto duplicou o dataset para ~12000 amostras e tornou o modelo muito mais robusto.

### A arquitectura da CNN

Usei 4 blocos convolucionais com o número de filtros a duplicar em cada camada (32→64→128→256). A ideia é que as primeiras camadas detectam padrões simples no espectrograma e as camadas mais profundas combinam esses padrões para reconhecer estruturas musicais mais complexas.

Adicionei BatchNormalization e Dropout em cada bloco para evitar overfitting, e usei GlobalAveragePooling2D no final em vez de Flatten para reduzir parâmetros.

### Majority voting na previsão

Quando o modelo prevê o género de uma música nova, analisa cada segmento de 5 segundos individualmente e no final combina as probabilidades de todos os segmentos. O género com maior probabilidade média vence. Isto torna o modelo mais robusto — um segmento ambíguo não estraga o resultado final.

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 95.82% |
| Macro F1-score | 0.958 |

O modelo atingiu resultados acima de 0.93 em todos os géneros excepto rock (0.92), que provavelmente confunde com blues e country por partilharem características sonoras semelhantes.

O treino parou automaticamente ao epoch 54 de 100 porque o early stopping detectou que o modelo não estava a melhorar — sinal de que convergiu bem sem precisar de mais epochs.