import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
#VARIABLES
T_room = 293,15 #kelvins
T_c1 = 293,15 #kelvins
T_c2 = 293,15 #kelvins
T_cc = 293,15 #kelvins
T_t = 293,15 #kelvins

PERTE_ET_GAIN = np.loadtxt('PerteEtGain.txt') # array constante du fichier texte 



def celcius_en_kelvin(Temp_celcius):
    '''fonction qui transforme une température en degés celcius en une température en degrés kelvin'''
    return (Temp_celcius + 273.15)

def Coeff_performance_chaud(temp_in, temp_out):
    '''calcule le coefficient de performance d'une pompe à chaleur
    TOUTES LES TEMPERATURES EN KELVINS 
    temp_in est la température extérieure (la plus froide)
    temp_out est la température intérieure (la plus chaude)'''
    return(0.5*temp_out/(temp_in - temp_out))

def Coeff_performance_froid(temp_in, temp_out):
    '''calcule le coefficient de performance d'une machine à froid'''
    return(0.5*temp_in/(temp_in - temp_out))

def T_optimale(T_room, T_surface):
    '''calcule la température ressentie en fonction de la chaleur de la pièce et celles des surfaces'''
    return((T_room+T_surface)/2)

def Efficacite(Pwr_in,Pwr_out):
    '''calcule l'efficacité d'une machine
    Pwr_in c'est la puissance fournie à la machine
    Pwr_out c'est le travail que fournit la machine'''
    return (Pwr_out/Pwr_in)

