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
    def __init__(self, num_input, epochs=1000):
        self.epochs = epochs
        self.neuroni = [Neurone(num_input, Sigmoide, dSigmoide) for _ in range(num_input)]
        self.neurone_output = Neurone(num_input, Sigmoide, dSigmoide)
        
    def FeedForward(self, inputs):
        outputs = [self.neuroni[i].FF(inputs) for i in range(len(self.neuroni))]
        
        output_finale = self.neurone_output.FF(np.array(outputs))
        
        return output_finale
    
    def BackPropagation(self, inputs, vatteso):
        outputs = [self.neuroni[i].FF(inputs) for i in range(len(self.neuroni))]

        final_output = self.neurone_output.FF(np.array(outputs))

        errore = final_output - vatteso
        
        _errori = self.neurone_output.BP(np.array(outputs), errore)
        
        for i, neurone in enumerate(self.neuroni):
            _ = neurone.BP(inputs, _errori[i])
        
        return errore **2
    
    def train(self, dataset):
        dataset = np.array(dataset)
        dataset = dataset / np.max(dataset, axis=0)
        for epoca in range(self.epochs):
            ErroreEpoca = 0
            for elemento in dataset:
                inputs = elemento[:-1]
                result = elemento[-1]
                
                ErroreEpoca += self.BackPropagation(inputs, result) **2
            
            ErroreEpoca = math.sqrt(ErroreEpoca/len(dataset))
            print(f"Epoca {epoca}, Errore: {ErroreEpoca}")
        
        return ErroreEpoca
            
    
        
DatasetCani = [[4,20,0],[5,24,0],[18,70,1],[20,80,1]] # Questo è il Dataset che è rappresentato come una lista di liste
DatasetGatti = [[4,20,1],[5,24,1],[18,70,0],[20,80,0]] # Questo è il Dataset che è rappresentato come una lista di liste
# La prima lista contiene le triple di dati che a loro volta solo una lista contenente il Peso, l'altezza e il risultato
# Il risultato è 0 nel caso di un gatto, 1 nel caso di un cane

rete_neurale_cani = NeuralNetwork(2, 1000)
rete_neurale_gatti = NeuralNetwork(2, 1000)

rete_neurale_cani.train(DatasetCani)
rete_neurale_gatti.train(DatasetGatti)
    
while True: # Creiamo un ciclo infinito: ANCHE SE NON SI FA MAI!
    peso = float (input ("Dammi il peso dell'animale: ")) / 20
    altezza = float (input ("Dammi l'altezza dell'animale: ")) / 80
    risultato_cane = rete_neurale_cani.FeedForward(np.array([peso, altezza]))
    risultato_gatto = rete_neurale_gatti.FeedForward(np.array([peso, altezza]))
    
    print(risultato_cane, risultato_gatto)