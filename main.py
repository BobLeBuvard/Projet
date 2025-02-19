import numpy as np
#VARIABLES
T_room = 293,15 #kelvins = 20 degrés Celcius
T_c1 = 293,15 #kelvins
T_c2 = 293,15 #kelvins
T_cc = 293,15 #kelvins
#test
T_t = 293,15 #kelvins
C_room = 1 # Capacité de la pièce régulée
C_c1 = 1 # Capacité de la partie supérieure béton
C_c2 = 1 # Capacité de la partie inférieure béton
R_s = 1 #Résistance de la surface entrre le béton et la pièce régulée
R_x = 1 #Résistance de contact entre les tubes & la partie centrale du béton

T = np.array([T_room, T_t, T_cc, T_c1,T_c2])
C = np.array([C_room, C_t, C_cc, C_c1,C_c2])

dT = np.zeros[5]
'''ici il y aura les cinq dérivées de la température, donc les cinq équations différentielles''' 
#CALCUL DE dT_cc
dT[2] = (1/C[2])*( (-1/(R_cc_moins_c1))*(T[2]-T[3])- (1/R_x)*(T[2]-T[1]) + (1/R_c2_moins_cc)*(T[4] - T[2]))

#CALCUL DE dT_c1 
dT[3] = (1/C_[3])*(-1/R_cc_moins_c1)*(T[3]-T[2])


#CALCUL DE dT_c2 
dT[4] = (1/C[4])* ((-1/R_c2_moins_cc)*(T[4]-T[2])+ (1/(R_r_moins_s)+ R_s_moins_c2)/(T[0] - T[4]))

#CALCUL DE dT_room
dT[0] = (1/C[0])*((-1/(R_r_moins_s +R_s_moins_c2))*(T[0]-T[4]+G(t)))  #Il manque la fonction G(t)
                  
#CALCUL DE dT_t
dT[1] = 1984 #PLACEHOLDER POUR LA VERITABLE EQUADIFF AVEC PARAMETRES


def celcius_en_kelvin(Temp_celcius):
    '''fonction qui transforme une température en degés celcius en une température en degrés kelvin'''
    return (Temp_celcius + 273.15)

def T_optimale(T_room, T_surface):
    '''calcule la température ressentie en fonction de la chaleur de la pièce et celles des surfaces'''
    return((T_room+T_surface)/2)


def calculTemperaturesIVP(FenetreDeTemperature, T0, rtol):
    
    return(t,T)