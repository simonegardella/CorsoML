Corso di Machine Learning:

- Lezione01 - Creazione di un neurone in grado di distinguere Cani e Gatti in base ad altezza (espressa in cm e peso espresso in Kg)
- Lezione02 - Rappresentiamo graficamente il loss (errore nell'addestramento)
- Lezione03 - Creazione della prima rete neurale con un layer di 2 perceptroni e 1 layer di output
- Lezione04 - Creazione di una rete con un numero di perceptroni e un numero di output variabili
- Lezione05 - Creazione di una rete con layer Hidden, con un numero di perceptroni, un numero di Neuroni nel layer Hidden e un numero di Classificatori variabili
- Lezione06 - Creazione di Classi di Dataset per comprendere come configurare la rete in funzione del dataset (un dataset con dati linearmente separabile [Tipo1], un dataset con dati a scacchiera [Tipo2])
- Lezione07 - ...prosecuzione della Lezione06 con la creazione di datasdet con dati concentrici [Tipo3] e un dataset con dati a spirale [Tipo4]
- Lezione08 - Realizzazione di una rete con Tensorflow in grado di processare tutti i dataset precedentemente creati e quello delle iris
- Lezione09 - Algoritmo di apprendimento per rinforzo (QLearn) e Rete Ricorrente sul testo de "I Promessi Sposi"
- Lezione10 - Realizzazione di una rete CNN per riconoscere i caratteri scritti a mano


Se si utilizza Anaconda:

 - Si crea il conda environment con il comando -> conda create -n *nomedellenvironment* python=3.12
 - Si attiva l'ambiente con -> conda activate *nomedellenvironment*

    +----------------------------------------------------------------+
    | Librerie da Installare:                                        |
    |                                                                |
    | matplotlib -> pip3 install matplotlib                          |
    | numpy -> pip3 install numpy                                    |
    | cv2 -> pip3 install opencv-python                              |
    | tensorflow -> pip3 install tensorflow                          |
    | tqdm -> pip3 install tqdm                                      |
    +----------------------------------------------------------------+

 - Per disattivare l'environment -> conda deactivate
 - Per visionare gli environment disponibili -> conda env list


Risorse:

Datasets -> Contiene i datasets usati nel corso
 - IPromessiSposti.txt -> testo de "I Promessi Sposi"
 - iris.csv -> elenco delle IRIS suddivise per larghezza e lunghezza del sepalo e del petalo
 - HandwrittenDigits -> Collezione di immagini di cifre scritte a mano 	


 Altre risorse:

 - Link al playground di Tensorflow : https://playground.tensorflow.org
 - Link a Tensorflow : https://www.tensorflow.org/?hl=it
 - Link a Keras : https://keras.io
 - Layers di Keras : https://keras.io/api/layers/
 - Funzioni di attivazione di Keras: https://keras.io/api/layers/activations/
