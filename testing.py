import numpy as np
from main import odefunction
from main import num_du_scenario
import scipy as scp
import SimTABS
import PerteEtGain
import matplotlib.pyplot as plt

#SCENARIO A MODIFIER DANS main.py
debug = True
#TESTER LES EXERCICES


T0 = [15, 15, 15, 15, 15]
FenetreDeTemperature = np.array([0, 24]) # fenetre de test comme demandé
h = 0.1  # pas de temps ( toutes les 6 minutes)

t, T = SimTABS.calculTemperaturesEuler(FenetreDeTemperature, T0, h)
if debug:  
    #print(T[1])
    plt.plot(t,T[0],label = 'T_room') 
    plt.plot(t,T[1],label = 'T_t') 
    plt.plot(t,T[2],label = 'T_cc') 
    plt.plot(t,T[3],label = 'T_c1') 
    plt.plot(t,T[4],label = 'T_c2')
    plt.show()



    
'''TEMPERATURE CORRIGEE -> affichage des graphes de température si debug = True, sinon mettre sur debug = False à la ligne 5 '''