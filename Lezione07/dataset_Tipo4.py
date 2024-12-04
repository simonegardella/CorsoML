import matplotlib.pyplot as plt
import numpy as np
import random
import math

class Dataset ():
    def __init__ (self, numerooutput, numeroelementi = 0):
        self.data = np.array([]) # E' una lista di liste di cui quella interna contiene: x,y,...
        self.numerooutput = numerooutput
        if numeroelementi > 0:
            self.Populate(numeroelementi)
        
    def Populate (self, numeropunti):
        data = []
        trim = 0.0
        puntiAngolo = int(numeropunti / 540)
        
        for angolo in range (540):
            trim += .1/540
            for classe in range (self.numerooutput):
                angolorad = (angolo+(360/self.numerooutput*classe))/180*math.pi
                for i in range (puntiAngolo):
                    raggiominimo = angolo/540*2-trim/2
                    raggiomassimo = angolo/540*2+trim/2
                    raggio = raggiominimo+random.random()* (raggiomassimo-raggiominimo)
                    x= raggio *math.cos(angolorad) + 2
                    y = raggio*math.sin(angolorad) + 2
                    resultset = [1 if t==classe else 0 for t in range(self.numerooutput)]
                    data.append ([x,y, *resultset] )
        self.data = np.array(data)
        
    def Shuffle (self):
        np.random.shuffle (self.data)
    def Normalizzazione (self):
        self.maxes = np.max(self.data,axis=0)
        self.data = self.data / self.maxes
        
    def Denormalizza (self):
        self.data = self.data * self.maxes
        
    def Dataset (self):
        return self.data[:,:2]
    def Resultset (self):
        return self.data[:,2:]
    
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
