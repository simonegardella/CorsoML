import numpy as np

stati = azioni = 24

peso_stare = 1
peso_muoversi = 0.8

Ricompense = np.zeros ((stati,azioni)) # Vincoli
Qualita = np.zeros_like (Ricompense) # Matrice da addestrare


def Stato (stato):
    return stato - 1

def Varco (stato1, stato2):
    global Ricompense
    Ricompense[Stato(stato1),Stato(stato2)] = 1
    Ricompense[Stato(stato2),Stato(stato1)] = 1
    
def ImpostaGoal (stato):
    global Ricompense
    
    # Azzeriamo il premio precedente
    for s in range(Ricompense.shape[0]):
        for a in range(Ricompense.shape[1]):
            if Ricompense[s,a] == 100:
                Ricompense[s,a] = 0
                
    # Impostiamo il nuovo premiopremio
    Ricompense[Stato(stato),Stato(stato)] = 100
    
def Bellman (stato,azione):
    global Ricompense, Qualita,peso_stare, peso_muoversi
    return peso_stare * Ricompense[stato,azione] + peso_muoversi * max (Qualita[azione,:])

def StampaMigliorpercorso (partenza = 1):
    global Qualita
    percorso = []
    perseguibile = True
    percorso.append (partenza)
    goal = np.max (Qualita)
    stato = partenza
    while Qualita[Stato(stato),Stato(stato)] < goal and perseguibile:
        
        stato = np.argmax (Qualita[Stato (stato),:])+1
        print (stato)
        perseguibile = not stato in percorso
        if perseguibile:
            percorso.append(stato)
            
    return percorso
        
    
Varco (1,2)
Varco (1,7)
Varco (2,8)
Varco (8,14)
Varco (14,15)
Varco (7,13)
Varco (13,19)
Varco (19,20)
Varco (20,21)
Varco (15,9)
Varco (15, 16)
Varco (9,3)
Varco (16,10)
Varco (16,22)
Varco (22,23)
Varco (23,24)
Varco (24,18)
Varco (18,17)
Varco (17,11)
Varco (11,12)
Varco (12,6)
Varco (6,5)
Varco (5,4)

Varco (3,4) # Varco addizionale per tentativi di dual path

ImpostaGoal(4)

def Addestramento(epoche = 100):
    global Ricompense,Qualita
    Qualita = np.zeros_like (Ricompense)
    for epoca in range (epoche):
        for stato in range (stati):
            for azione in range (azioni):
                if Ricompense[stato,azione]!=0:
                    Qualita[stato,azione] = Bellman (stato,azione)
                    
                    