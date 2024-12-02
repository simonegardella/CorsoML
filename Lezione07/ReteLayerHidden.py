import random
import math
import numpy as np
import matplotlib.pyplot as plt
from dataset_Tipo2 import *

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
    def __init__(self, num_input, num_hidden, num_output, epochs=1000):
        self.epochs = epochs
        self.neuroni_input = [Neurone(num_input, Sigmoide, dSigmoide) for _ in range(num_input)]
        self.neuroni_hidden = [Neurone (num_input, Sigmoide, dSigmoide) for _ in range(num_hidden)]
        self.neuroni_output =[Neurone(num_hidden, Sigmoide, dSigmoide) for _ in range(num_output)]
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.history=[]
        
    def FeedForward(self, inputs):
        outputs_neuroni_input = [self.neuroni_input[i].FF(inputs) for i in range(len(self.neuroni_input))]
        outputs_neuroni_hidden = [self.neuroni_hidden[i].FF(outputs_neuroni_input) for i in range(len(self.neuroni_hidden))]
        output_finale = np.zeros(self.num_output)
        for i in range (self.num_output):
            output_finale[i] = self.neuroni_output[i].FF(np.array(outputs_neuroni_hidden))

        return output_finale
    
    def Softmax (elements):
        return np.exp(elements) / sum (np.exp(elements))
    
    def BackPropagation(self, inputs, vattesi):
        outputs_neuroni_input = [self.neuroni_input[i].FF(inputs) for i in range(len(self.neuroni_input))]
        outputs_neuroni_hidden = [self.neuroni_hidden[i].FF(outputs_neuroni_input) for i in range(len(self.neuroni_hidden))]
        output_finale = np.zeros(self.num_output)
        for i in range (self.num_output):
            output_finale[i] = self.neuroni_output[i].FF(np.array(outputs_neuroni_hidden))


        errori = output_finale - vattesi
        
        erroriintermedi_hidden = np.zeros (self.num_hidden)
        for i in range (self.num_output):
            erroriintermedi_hidden = erroriintermedi_hidden + self.neuroni_output[i].BP(np.array(outputs_neuroni_hidden), errori[i])
        
        erroriintermedi_input = np.zeros (self.num_input)
        for i in range (self.num_hidden):
            erroriintermedi_input = erroriintermedi_input + self.neuroni_hidden[i].BP(np.array(outputs_neuroni_input), erroriintermedi_hidden[i])
        
        for i, neurone in enumerate(self.neuroni_input):
            _ = neurone.BP(inputs, erroriintermedi_input[i])
        
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
       
       
ClasseDataset = Dataset(3,100)
ClasseDataset.Show()

Dataset = ClasseDataset.data 

Rete = NeuralNetwork(2,15,3,10000) 
# Il numero di neuroni hidden deve ragionevolmente essere compreso tra 2n +1 rispetto al layer adiacente
# Noi abbiamo scelto 7

Rete.train(Dataset)


plt.figure()
plt.plot (range(len(Rete.history)),Rete.history)

plt.show()


