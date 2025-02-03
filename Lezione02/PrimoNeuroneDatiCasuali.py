import math # Serve per le funzioni matematiche
import random # Serve per generare numeri casuali
from matplotlib import pyplot as plt
import numpy as np


Dataset = [[4,20,0],[5,24,0],[18,70,1],[20,80,1]] # Questo è il Dataset che è rappresentato come una lista di liste
# La prima lista contiene le triple di dati che a loro volta solo una lista contenente il Peso, l'altezza e il risultato
# Il risultato è 0 nel caso di un gatto, 1 nel caso di un cane


massimoPeso = 20    # Massimo valore della prima caratteristica (Peso in Kg)
massimaAltezza = 80 # Massimo valore della seconda caratteristica  (Altezza in cm)

thr = 0.1 # Imposto la Threshold

random.shuffle(Dataset) # Questa funzione consente di mescolare i dati del Dataset


w1= random.random()
w2 = random.random()
bias = random.random()
lr = 0.05

def Sig (x): # Sigmoide (x)
    return 1/(1+math.exp(-x))

def dSig (x): # Derivata del Sigmoide(x)
    t = Sig (x)
    return t*(1-t)


def FNeurone (Peso, Altezza): # Funzione del neurone
    global w1,w2,bias
    return Peso*w1+Altezza*w2+ bias

def Output (Peso,Altezza): # Output del neurone passando per la funzione di attivazione Sigmoide che abbiamo definito (FeedForward)
    return Sig (FNeurone(Peso,Altezza))

def BackPropagation (Peso,Altezza, ValoreAtteso): # Addestrare il neurone passando come parametri il peso, l'altezza e il valore atteso
    global w1,w2,bias,lr # Utilizzare variabili che non sono presenti nella funzione ma nel codice globale
    Out = Output(Peso,Altezza) # Il valore di output del neurone
    FN = FNeurone(Peso,Altezza) # Il valore nel neurone senza la funzione di attivazione
    Errore = (Out - ValoreAtteso)**2 # Calcolo l'errore grazie al fatto che ho il valore atteso corretto
    dW1 = 2 * (Out - ValoreAtteso) * dSig(FN) * Peso
    dW2 = 2 * (Out - ValoreAtteso) * dSig(FN) * Altezza
    dBias = 2 * (Out - ValoreAtteso) * dSig(FN) * 1
    w1 = w1 - dW1 * lr
    w2 = w2 - dW2 * lr
    bias = bias - dBias * lr
    return Errore
    

#Addestriamo la nostra rete
for epoca in range (1500):  #Scorro il range (un insieme di valori da 0 a 4999) e li assegno alla variabile epoca
    ErroreEpoca = 0 # Azzero per l'epoca in corso la somma dell'errore quatratico
    for elemento in Dataset: 
        peso = elemento[0] / massimoPeso # Normalizziamo rispetto al peso
        altezza = elemento[1] /massimaAltezza # Normalizziamo rispetto all'altezza
        vatteso = elemento[2]
        ErroreEpoca = ErroreEpoca + BackPropagation(peso,altezza,vatteso)
    ErroreEpoca = math.sqrt(ErroreEpoca/len (Dataset)) # Calcolo l'errore medio divendo la somma precedente per il numero di valori e applicando la radice quadrata.
    print (epoca, ErroreEpoca) # Per ogni epoca stampo l'errore

    
    
cani = []
gatti =[]
noanimals = []

while len (cani) <100 and len (gatti) <100:
    peso = random.random()
    altezza = random.random()
    #print (f"L'animale di peso {peso*massimoPeso} e altezza {altezza*massimaAltezza} ", end="")
    risultato = Output (peso,altezza)
    if risultato < thr: # Se è sotto thr allora diciamo che è un gatto
        #print (f"E' un gatto ({risultato})!!!")
        gatti.append([peso,altezza])
    elif risultato > 1-thr:  # Se è sopra 1- thr è un cane
        #print (f"E' un cane {risultato}!!!")
        cani.append([peso,altezza])
    else: # Altrimenti non sappiamo di che animale si tratta
        #print (f"Purtroppo non so di che animale si tratta! {risultato}")
        noanimals.append([peso,altezza])
    
    

cani = np.array(cani)
gatti = np.array (gatti)
noanimals = np.array (noanimals)

print (cani, gatti)

plt.figure()
plt.scatter(cani[:,0], cani[:,1])
plt.scatter(gatti[:,0], gatti[:,1])
plt.scatter(noanimals[:,0], noanimals[:,1])
plt.show()
    
    
    


    