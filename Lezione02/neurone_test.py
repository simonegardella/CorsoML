import math
import random

import numpy as np
from matplotlib import pyplot as plt

dataset: list[list] = [[3, 1], [4, 1], [15, 0], [17, 0]] #Differenziare le angurie (etichetta 0) dai piselli (etichetta 1) in base al loro diametro

max_diametro: int = 20

threshold: float = 0.2

random.shuffle(dataset)

bias: float = random.random()
wheight_1: float = random.random()

learning_rate: float = 0.05

def sigmoid(x: float) -> float:
    return 1/(1+math.exp(-x))

def d_sig(x: float) -> float:
    t = sigmoid(x)
    return t*(1-t)

def funzione_neurone(x: float) -> float:
    global wheight_1, bias
    
    return (wheight_1 * x) + bias

def feed_forward(x: float) -> float:
    return sigmoid(funzione_neurone(x))

def back_propagation(diametro: float, valore_atteso: float) -> float:
    global wheight_1, learning_rate, bias
    
    output: float = feed_forward(diametro)
    output_funzione: float = funzione_neurone(diametro)
    
    loss: float = (output-valore_atteso)**2
    
    d_weight_1: float = 2 * (output - valore_atteso) * d_sig(output_funzione) * diametro
    d_bias: float = 2 * (output - valore_atteso) * d_sig(output_funzione)
    
    wheight_1 = wheight_1 - (d_weight_1 * learning_rate)
    bias = bias - (d_bias * learning_rate)
    
    return loss

for epochs in range(500):
    epoch_loss: float = 0
    
    for element in dataset:
        diametro = element[0] / max_diametro
        label = element[1]
        
        epoch_loss = epoch_loss + back_propagation(diametro, label)
        
    epoch_loss = math.sqrt(epoch_loss/len(dataset))
    
    # print(f"EPOCA --> {epochs} \t LOSS --> {epoch_loss}")
    
angurie: list = []
piselli: list = []
anything: list = []
    
diametro_input: int = int(input("Dammi la larghezza che desideri verificare: "))

risultato = feed_forward(diametro_input)

print(f"Risultato --> {risultato}")
