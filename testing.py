import numpy as np
from main import odefunction
from main import num_du_scenario
import scipy as scp
import SimTABS
import PerteEtGain
import matplotlib.pyplot as plt

days = 20 #calculer sur 1 jour par exemple

#SCENARIO A MODIFIER DANS main.py
debug = True


#TESTER LES EXERCICES


T0 = np.array([15, 15, 15, 15, 15])
FenetreDeTemps = np.array([0, 24]) # fenetre de test comme demandé
h = 0.1  # pas de temps ( toutes les 6 minutes)

#t, T = SimTABS.calculTemperaturesEuler(FenetreDeTemperature, T0, h)

t,T = SimTABS.calculCycles(5,T0,FenetreDeTemps,h)
if debug:  
    #print(T[1])
    plt.plot(t,T[0],label = 'T_room') 
    plt.plot(t,T[1],label = 'T_t') 
    plt.plot(t,T[2],label = 'T_cc') 
    plt.plot(t,T[3],label = 'T_c1') 
    plt.plot(t,T[4],label = 'T_c2')
    plt.show()

