import random
import math
import numpy as np
import matplotlib.pyplot as plt

def Sigmoide (x):
    return 1/ (1+math.exp (-x))

def dSigmoide (x):
    t = Sigmoide(x)
    return  t * (1-t)

def Linear (x):
    return x

def dLinear (x):
    return 1

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
        
    def FF (self, inputs): # Feed Forward
        return self.fa (self.FN(inputs))
    
    def BP (self, inputs, errore): # Back Propagation
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

class NeuralNetwork():
    def __init__(self, num_input, num_output, epochs=1000):
        self.epochs = epochs
        self.neuroni = [Neurone(num_input, Sigmoide, dSigmoide) for _ in range(num_input)]
        self.neuroni_output =[Neurone(num_input, Sigmoide, dSigmoide) for _ in range(num_output)]
        self.num_input = num_input
        self.num_output = num_output
        self.history=[]
        
    def FeedForward(self, inputs):
       
        
        output_finale = np.zeros(self.num_output)
        for i in range (self.num_output):
            output_finale[i] = self.neuroni_output[i].FF(np.array(outputs))


        # Meccanismo della Temperatura
        output = np.exp(np.log (output_finale)*3)
        output = np.zeros_like (output_finale)
        
        # Identificazione del migliore
        
        output[np.argmax(output_finale)] = 1
        
        return output
    
    def Softmax (elements):
        return np.exp(elements) / sum (np.exp(elements))
    
    def BackPropagation(self, inputs, vattesi):
        outputs = [self.neuroni[i].FF(inputs) for i in range(len(self.neuroni))]

        output_finale = np.zeros(self.num_output)
        for i in range (self.num_output):
            output_finale[i] = self.neuroni_output[i].FF(np.array(outputs))

        errori = output_finale - vattesi
        
        erroriintermedi = np.zeros (self.num_input)
        for i in range (self.num_output):
            erroriintermedi = erroriintermedi + self.neuroni_output[i].BP(np.array(outputs), errori[i])
        
        for i, neurone in enumerate(self.neuroni):
            _ = neurone.BP(inputs, erroriintermedi[i])
        
        return errori **2
    
    def bilanciaDS (self,dataset):
        shape = dataset.shape
        DS = np.zeros((int(shape[0]*2/3),shape[1]))
        
        casi = []
        
        for i in range (len(dataset)):
            if dataset[i,2]==1:
                casi.append(i)
            
        while len (casi) < shape[0]*2/3:
            valorecasuale = random.randint (0,shape[0]-1)
            if not valorecasuale in casi and dataset[valorecasuale,2] != 1:
                casi.append(valorecasuale)
                    
        for i, caso in enumerate(casi):
            DS[i,:] = dataset[caso,:]
            
        return DS
    
    def train(self, dataset):
        dataset = np.array(dataset)
        
        dataset = dataset / np.max(dataset, axis=0)
        for epoca in range(self.epochs):
            ErroreEpoca = np.zeros(3)
            
            dataset_temp = dataset.copy()
            #dataset_temp = self.bilanciaDS(dataset)
            #np.random.shuffle (dataset_temp)
            for elemento in dataset_temp:
                inputs = elemento[:2]
                result = elemento[2:]
                
                ErroreEpoca = ErroreEpoca + self.BackPropagation(inputs, result) **2
            
            ErroreEpoca = (ErroreEpoca/len(dataset)) ** 0.5
            print(f"Epoca {epoca}, Errore: {ErroreEpoca}")
            self.history.append(ErroreEpoca)
        
        return ErroreEpoca
            
    
#       4       5       18      20      250         280        
#       20      24      70      80      420         520
#       Gatto   Gatto   Cane    Cane    Cavallo     Cavallo
#
# La prima lista contiene le triple di dati che a loro volta solo una lista contenente il Peso, l'altezza e il risultato
# Il risultato è 0 nel caso di un gatto, 1 nel caso di un cane
# DatasetCani =   [[4,20,0],[5,24,0],[5, 21,0],[18,70,1],[20,80,1],[19,60,1],[420,250,0],[500,280,0],[480,265,0]] # Questo è il Dataset che è rappresentato come una lista di liste
# DatasetGatti =  [[4,20,0],[5,24,0],[5, 21,0],[18,70,1],[20,80,1],[19,60,1],[420,250,0],[500,280,0],[480,265,0]] # Questo è il Dataset che è rappresentato come una lista di liste
# DatasetCavalli =[[4,20,0],[5,24,0],[5, 21,0],[18,70,1],[20,80,1],[19,60,1],[420,250,0],[500,280,0],[480,265,0]] # Questo è il Dataset che è rappresentato come una lista di liste

# Il Dataset deve essere "aggiustato". Gli aggiustamenti possibili sono:
# a) Aggiungere elementi al DS (aggiungiamo 1 cavallo, 1 gatto e 1 cane a tutti e tre i DS)
# b) Eliminiamo due negativi per ogni DS


# Percorriamo l'ipotesi A

       
Dataset =   np.array([[4,20,1,0,0],[5,24,1,0,0],[5, 21,1,0,0],[18,70,0,1,0],[20,80,0,1,0],[19,60,0,1,0],[420,250,0,0,1],[500,280,0,0,1],[480,265,0,0,1]]) # Questo è il Dataset che è rappresentato come una lista di liste

Rete = NeuralNetwork(2,3,1000)

Rete.train(Dataset)

"""
plt.figure()
plt.plot (range(len(Rete.history)),Rete.history)

plt.show()
"""


while True: # Creiamo un ciclo infinito: ANCHE SE NON SI FA MAI!
    peso = float (input ("Dammi il peso dell'animale: ")) / np.max(Dataset, axis=0)[0]
    altezza = float (input ("Dammi l'altezza dell'animale: ")) / np.max(Dataset, axis=0)[1]
    
    risultati = Rete.FeedForward(np.array([peso,altezza]))
    
    print("Gatto: {:.2f}, Cane: {:.2f}, Cavallo: {:.2f}".format(*risultati))