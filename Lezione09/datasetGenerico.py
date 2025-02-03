import matplotlib.pyplot as plt
import numpy as np
import random

class Dataset ():
    def __init__ (self,dati, lunghezza=30,pred=1):
        data = []# E' una lista di liste di cui quella interna contiene: x,y,...
        self.lunghezza = lunghezza
        
        for inizio in range (len(dati)- (lunghezza+pred)):
            datatemp = dati[inizio:inizio + (lunghezza+pred)]
            for i in range (len (datatemp)):
                t = np.zeros (27)
                t[datatemp [i]] = 1
                datatemp[i] = t.copy()            
            data.append (datatemp)
            
        self.data= np.array(data)
        
    def Normalizzazione (self):
        self.maxes = np.max(self.data,axis=0)
        self.data = self.data / self.maxes
        
    def Denormalizza (self):
        self.data = self.data * self.maxes
        
    def Dataset (self):
        return self.data[:,:self.lunghezza]
    def Resultset (self):
        return self.data[:,self.lunghezza:]
    