import tensorflow as tf
import numpy as np
from dataset_iris import *
from matplotlib import pyplot as plt

datasetIris = Dataset(4)

datasetIris.Shuffle()
datasetIris.Normalizzazione()

# Questo modello lavora con 4 input e 3 output

try :
    modello = tf.keras.models.load_model("Iris.keras")
except:
    print ("Il modello non esiste, quindi lo creo!!!!")
    modello = tf.keras.models.Sequential(
        [
            tf.keras.Input (shape=datasetIris.Dataset().shape[1:]),
            tf.keras.layers.Dense (4,"sigmoid"),
            tf.keras.layers.Dense (8,"sigmoid"),
            tf.keras.layers.Dense (3,"softmax")
        ]
        )
    


    modello.compile (optimizer=tf.keras.optimizers.Adam(learning_rate=.002),
                    loss = "mse")

    modello.summary()

    modello.fit (datasetIris.Dataset(),datasetIris.Resultset(),epochs =1000)
    
    plt.figure()
    plt.plot (range(len(modello.history.history['loss'])),modello.history.history['loss'])
    plt.show()
    
    

    modello.save ("Iris.keras")


modello.summary()

output = modello.predict (datasetIris.Dataset())
thr = 0.5
for indice in range (output.shape[0]):
    previsione =  output[indice]-datasetIris.Resultset()[indice]
    print (indice, sum(previsione))
