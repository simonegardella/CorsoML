import matplotlib.pyplot as plt
import numpy as np
import random

class Dataset ():
    def __init__ (self, numerooutput, numeroelementi = 0):
        self.data = np.array([]) # E' una lista di liste di cui quella interna contiene: x,y,...
        self.numerooutput = numerooutput
        if numeroelementi > 0:
            self.Populate(numeroelementi)
        
    def Populate (self, numeropunti):
        data = []
        for x in range (numeropunti):
            temp = random.randint(0,self.numerooutput-1) 
            resultset = [1 if t==temp else 0 for t in range(self.numerooutput)]
            
            data.append ([random.random ()*4, random.random()*4, *resultset] )
        self.data = np.array(data)
        
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
