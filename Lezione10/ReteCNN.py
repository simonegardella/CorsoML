import tensorflow as tf
import numpy as np
import cv2 
import os
import random
import tqdm

cifre = [str(x) for x in range (10)]

Dataset = []
Resultset = []


for cifra in cifre:
    for data in os.walk (f'../Datasets/HandwrittenDigits/{cifra}/'):
        for file in data[-1]:
            immagine = cv2.imread (f'../Datasets/HandwrittenDigits/{cifra}/{file}')
            immagine = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)/255
            immagine = immagine.reshape((*immagine.shape,1))

            Dataset.append (immagine)
            out = [0]* 10
            out[int(cifra)] = 1
            Resultset.append (out)
            
            


print ("shuffling...")
for k in tqdm.tqdm(range (int(len(Dataset)*50/100))):
    t1 = random.randint(0,len(Dataset)-1)
    t2 = random.randint(0,len(Dataset)-1)
    dTemp = Dataset[t1].copy()
    rTemp = Resultset[t1].copy()
    Dataset[t1] = Dataset[t2].copy()
    Resultset[t1] = Resultset[t2].copy()
    Dataset[t2] = dTemp
    Resultset[t2] = rTemp

Dataset = np.array(Dataset)
Resultset = np.array(Resultset)

try :
    modello = tf.keras.models.load_model("digits.keras")
except:
    print ("Il modello non è stato salvato quindi lo creo da Zero")
    modello = tf.keras.models.Sequential(
    [
    tf.keras.Input (shape = Dataset.shape[1:]),
    tf.keras.layers.Conv2D(64,2,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,padding="same"),
    tf.keras.layers.Conv2D(128,2,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,padding="same"),
    tf.keras.layers.Conv2D(256,2,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,padding="same"),
    tf.keras.layers.Flatten(),
     tf.keras.layers.Dense (500),
    tf.keras.layers.Dense (100),
    tf.keras.layers.Dense (Resultset.shape[1])

    ]
    
)

modello.compile  (optimizer=tf.keras.optimizers.Adam(),
                loss = "mse")
modello.fit (Dataset,Resultset, epochs=1)
modello.save('digits.keras')

modello.summary()
immagine = np.zeros((56,56),dtype=np.uint8)
for i, dato in enumerate(Dataset[:4000]):
    
    output = modello.predict(np.array([dato]),verbose=0)
    risultato_Resultset = np.argmax(Resultset[i])
    risultato_Modello = np.argmax(output[0])
    if risultato_Modello == risultato_Resultset:
        print ('.',end='',flush=True)
    else:
        print (risultato_Resultset,end='',flush=True)
        immagine = np.zeros((56,56),dtype=np.uint8)
        for y in range (dato.shape[0]):
            for x in range (dato.shape[1]):
                immagine[y+27,x+27] = int(dato[y,x]*255)
        cv2.putText(immagine, f"{risultato_Modello}", (28,28), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1, cv2.LINE_AA) # Normale la previsone della rete
        cv2.putText(immagine, f"{risultato_Resultset}", (0,56), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA) # In grassetto il valore corretto
                
    for y in range (dato.shape[0]):
        for x in range (dato.shape[1]):
            immagine[y,x] = int(dato[y,x]*255)
    cv2.imshow ("Finestra",immagine)
    cv2.waitKey(1)
    
    


            



