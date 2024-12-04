import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv

class Dataset ():
    def __init__ (self, numerooutput, numeroelementi = 0):
        self.data = np.array([]) # E' una lista di liste di cui quella interna contiene: x,y,...
        self.numerooutput = numerooutput
        self.classificatore = {}
        self.classificatore['Setosa'] = [1,0,0]
        self.classificatore['Versicolor'] = [0,1,0]
        self.classificatore['Virginica'] = [0,0,1]
        
        self.Populate(numeroelementi)

        
    def Populate (self, numeropunti):
        f=open ('../Datasets/iris.csv')
        
        data = []
        csvreader = csv.reader(f.readlines(),delimiter=',',quotechar='"')
        for i,riga in enumerate(csvreader):
            if i >0:
                # ['5.1', '3.5', '1.4', '.2', 'Setosa']         ==> [5.1,3.5,1.4,0.2,1,0,0] 
                # ['5.2', '2.7', '3.9', '1.4', 'Versicolor']    ==> [5.2,2.7,3.9,1.4,0,1,0] 
                # ['6.7', '3.3', '5.7', '2.5', 'Virginica']     ==> [6.7,3.3,5.7,2.5,0,0,1]
                riga_adeguata = [*[float(x) for x in riga[:4]],*self.classificatore[riga[4]]]
                data.append (riga_adeguata)
                
        f.close()
        self.data = np.array(data)
        
        
    def Shuffle (self):
        np.random.shuffle (self.data)
    
    def Normalizzazione (self):
        self.maxes = np.max(self.data,axis=0)
        self.data = self.data / self.maxes
        
    def Denormalizza (self):
        self.data = self.data * self.maxes

        
    def Dataset (self):
        return self.data[:,:4]
    def Resultset (self):
        return self.data[:,4:]
    
    def Show (self):
        plt.figure()
        for classe in range(0,self.numerooutput):
            ClasseX = []
            for dato in self.data:
                if dato[classe+2] == 1.: # Il valore che indica la classe in esame è il valore alla posizione 2 [x,y]+ classe  
                    ClasseX.append(dato[:2]) # Prendo solo x,y
                
            ClasseX = np.array (ClasseX)
            if ClasseX.shape[0]>0:
                plt.scatter(ClasseX[:,0], ClasseX[:,1])
                
        
        plt.show()
