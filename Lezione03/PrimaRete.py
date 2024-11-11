import random
import math
import numpy as np

def Sigmoide (x):
    return 1/ (1+math.exp (-x))

def dSigmoide (x):
    t = Sigmoide(x)
    return  t * (1-t)

class Neurone():
    def __init__ (self, n_input,fa = lambda x:x, dfa= lambda x:1): # Costruttore della classe ovvero il metodo che viene invocato quando l'oggeto si crea!
        self.n_input = n_input
        self.w = np.random.random(size= n_input)
        self.b = random.random()
        self.fa = fa
        self.dfa = dfa
        self.lr = 0.1
        
    def FN (self, inputs):
        return sum (inputs*self.w)+ self.b
        
    def FF (self, inputs):
        return self.fa (self.FN(inputs))
    
    def BP (self, inputs, errore):
        fn = self.FN(inputs)
        # errore = (inputs- Vatt) ** 2
        
        # Calcoliamo le derivate di ciascun ramo del neurone e rispetto al suo bias
        dw = 2*errore*self.dfa(fn)* inputs * self.lr
        db = 2*errore*self.dfa(fn) * self.lr
        
        # Calcoliamo la parte di derivata da passare ai neuroni retrostanti
        errori = 2 * errore * self.dfa(fn) * self.w
        
        # Addestramento
        self.w = self.w -dw
        self.b -= db
        return errori
    
    
    
Dataset = [[4,20,0],[5,24,0],[18,70,1],[20,80,1]] # Questo è il Dataset che è rappresentato come una lista di liste
# La prima lista contiene le triple di dati che a loro volta solo una lista contenente il Peso, l'altezza e il risultato
# Il risultato è 0 nel caso di un gatto, 1 nel caso di un cane


massimoPeso = 20    # Massimo valore della prima caratteristica (Peso in Kg)
massimaAltezza = 80 # Massimo valore della seconda caratteristica  (Altezza in cm)


N1 = Neurone(2,Sigmoide,dSigmoide)
N2 = Neurone(2,Sigmoide,dSigmoide)
N3 = Neurone(2,Sigmoide,dSigmoide)


def FeedForward (inputs):
    Output1 = N1.FF (inputs)
    Output2 = N2.FF (inputs)
    output = N3.FF (np.array ([Output1,Output2]))
    return output

def BackPropagation (inputs, vatteso):
    Output1 = N1.FF (inputs)
    Output2 = N2.FF (inputs)
    output = N3.FF (np.array ([Output1,Output2]))
    errore = output- vatteso
    _errori = N3.BP (np.array ([Output1,Output2]), errore)
    
    _ = N1.BP (inputs,_errori[0])
    _ = N2.BP (inputs,_errori[1])
    
    return errore **2
    




for epoca in range (50):  #Scorro il range (un insieme di valori da 0 a 4999) e li assegno alla variabile epoca
    ErroreEpoca = 0 # Azzero per l'epoca in corso la somma dell'errore quatratico
    for elemento in Dataset: 
        peso = elemento[0] / massimoPeso # Normalizziamo rispetto al peso
        altezza = elemento[1] /massimaAltezza # Normalizziamo rispetto all'altezza
        vatteso = elemento[2]        
        inputs = np.array ([altezza,peso])
        ErroreEpoca += BackPropagation(inputs,vatteso) **2
    ErroreEpoca = math.sqrt(ErroreEpoca/len (Dataset)) # Calcolo l'errore medio divendo la somma precedente per il numero di valori e applicando la radice quadrata.
    print (epoca, ErroreEpoca) # Per ogni epoca stampo l'errore

