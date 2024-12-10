import numpy as np
import tensorflow as tf
from datasetGenerico import *

lunghezzafrase = 50
prev = 1
PromessiSposi = []
f = open ('../Datasets/IPromessiSposi.txt')
file = f.read()[:25000]
f.close()
for carattere in file.upper():
    if carattere == ' ':
        PromessiSposi.append (0)
    if (ord(carattere)>= 65 and ord (carattere) <= 90):
        PromessiSposi.append (ord (carattere)-64)

dataset = Dataset (PromessiSposi,lunghezzafrase,prev) 


try :
    modello = tf.keras.models.load_model("promessisposi.keras")
except:
    print ("Il modello non esiste, quindi lo creo!!!!")
    modello = tf.keras.models.Sequential(
        [
            tf.keras.Input (shape=dataset.Dataset().shape[1:]),
            tf.keras.layers.LSTM(dataset.Dataset().shape[1]*dataset.Dataset().shape[2], activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(dataset.Resultset().shape[1]*dataset.Resultset().shape[2]*2, activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(dataset.Resultset().shape[1]*dataset.Resultset().shape[2], activation='tanh'),
            tf.keras.layers.Reshape(dataset.Resultset().shape[1:]),
            tf.keras.layers.Dense (dataset.Resultset().shape[-1],"softmax")
        ]
        )



modello.compile (optimizer=tf.keras.optimizers.Adam(),
                loss = "mse")

modello.summary()
if False:
    with tf.device('/GPU:0'):   
        modello.fit (dataset.Dataset(),dataset.Resultset(),epochs =10)



modello.save ("promessisposi.keras")

frase_iniziale = "era una volta"
frase_iniziale = frase_iniziale[-lunghezzafrase:]
while len (frase_iniziale) < lunghezzafrase:
    frase_iniziale = " "+ frase_iniziale
    
first_input =[]
for  carattere in frase_iniziale.upper():
    if carattere == ' ':
        t = np.zeros (27)
        t[0] = 1
        first_input.append(t.copy())
    
    if (ord(carattere)>= 65 and ord (carattere) <= 90):
        c = ord (carattere)-64
        t = np.zeros (27)
        t[c] = 1
        first_input.append(t.copy())
        
print (first_input)
    

for t in range (50):
    print ("Input:",frase_iniziale[-lunghezzafrase:])
    to_text = ""
    with tf.device('/GPU:0'):  
        output = modello.predict(np.reshape(np.array(first_input),(1,lunghezzafrase,27)))
    for lettera in output[0]:
        first_input.append(lettera)
        mp = np.argmax(lettera)
        if mp == 0:
            to_text += " "
        else:
            to_text += chr(64+mp)
    frase_iniziale+=to_text
    first_input = first_input[-lunghezzafrase:]
    
print (frase_iniziale)
    