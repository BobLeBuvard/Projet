import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
#VARIABLES
T_room = 293,15 #kelvins = 20 degrés Celcius
T_c1 = 293,15 #kelvins
T_c2 = 293,15 #kelvins
T_cc = 293,15 #kelvins
T_t = 293,15 #kelvins
C_room = 1 # pièce régulée
C_c1 = 1 # partie supérieure béton
C_c2 = 1 # partie inférieure béton
s = 1 #surface entrre le béton et la pièce régulée
x = 1 #contact entre les tubes & la partie centrale du béton

T = np.array([T_room, T_t, T_cc, T_c1,T_c2])





def celcius_en_kelvin(Temp_celcius):
    '''fonction qui transforme une température en degés celcius en une température en degrés kelvin'''
    return (Temp_celcius + 273.15)

def T_optimale(T_room, T_surface):
    '''calcule la température ressentie en fonction de la chaleur de la pièce et celles des surfaces'''
    return((T_room+T_surface)/2)


def calculTemperaturesIVP(FenetreDeTemperature, T0, rtol):
    
    return(t,T)