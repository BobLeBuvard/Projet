#DEPRECATED: NE PLUS UTILISER LES FONCTIONS QUI SONT ICI SAUF POUR DES TESTS

import numpy as np
''' 
module qui contient toutes les variables. Pour l'instant ça charge tout dans la mémoire, on verra pour changer si on veut juste charger certaines parties pour certaines fonctions par la suite

'''

def kelvin(temp):
    return (temp+273.15) 
'''variables qu'on change pour le testing'''


FenetreDeTemps = np.array([0, 24]) # fenetre de test comme demandé -> taille du cycle
num_du_scenario = 1 # scénario 5 = scénario debug
h = 0.01  # pas de temps ( toutes les 6 minutes)
debug = True
T0 = kelvin(np.array([15,15,15,15,15])) #conitions initiales données -> ici mises en array en kelvins
delta_t = 0


''' variables assez fixes'''

default_tol = 10e-10 #choix arbitraire

#FORME DE l'array T 

# T = np.array([T_room, T_t, T_cc, T_c1,T_c2])

C_room = 12 # Capacité de la pièce régulée (kJ/m²K)
C_c1 = 50# Capacité de la partie supérieure béton (kJ/m²K)
C_c2 = 10# Capacité de la partie inférieure béton (kJ/m²K)
C_cc = 50 # Capacité de la partie centrale béton (kJ/m²K)
C_w = 30 # Capacité de l'eau (kJ/m²K)
C = np.array([C_room, C_w, C_cc, C_c1,C_c2, ])*1000 # kJ/m²K -> J/m²K


R_x = 0.025 #Résistance de contact entre les tubes & la partie centrale du béton (m²K/W)
R_w = 0.15 #Résistance de l'eau
R_cc_moins_c1 = 0.05
R_c2_moins_cc = 0.02
R_r_moins_s = 0.1
R_s_moins_c2 = 0.183