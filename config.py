#Ressuscité

import numpy as np
''' 
module qui contient toutes les variables. Pour l'instant ça charge tout dans la mémoire, on verra pour changer si on veut juste charger certaines parties pour certaines fonctions par la suite

'''


#CONFIG
FenetreDeTemps = np.array([0, 24]) # fenetre de test comme demandé -> taille du cycle
h = 0.01  # pas de temps ( toutes les 6 minutes)
T0 = np.array([15,15,15,15,15])  #conitions initiales données 
default_tol = 10e-10 #choix arbitraire
searchInterval = [0,20] # pour le delta_t qui varie entre 4h et 24h donc delta_t entre 0 et 20h
tol_rac = h
tol_temp = 0.01
max_jours = 30
num_du_scenario = 1
#FORME DE l'array T 

# T = np.array([T_room, T_t, T_cc, T_c1,T_c2])

C_room = 12 # Capacité de la pièce régulée (kJ/m²K)
C_c1 = 50# Capacité de la partie supérieure béton (kJ/m²K)
C_c2 = 10# Capacité de la partie inférieure béton (kJ/m²K)
C_cc = 50 # Capacité de la partie centrale béton (kJ/m²K)
C_w = 30 # Capacité de l'eau (kJ/m²K)
C = np.array([C_room, C_w, C_cc, C_c1,C_c2])*1000 # kJ/m²K -> J/m²K
inv_C = 1/C

R_x = 0.025 #Résistance de contact entre les tubes & la partie centrale du béton (m²K/W)
R_w = 0.15 #Résistance de l'eau
R_cc_moins_c1 = 0.05
R_c2_moins_cc = 0.02
R_r_moins_s = 0.1
R_s_moins_c2 = 0.183

debug = True