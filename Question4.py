from SimTABS import calculTemperaturesEuler
import RechercheRacine 
import numpy as np
from SimTabsFinal import calculTemperaturesEuler

#______________________________________________________________________________________________________#
# question 4.1



def T_optimale(T_room, T_surface):
    """Calcule la température de confort selon la norme"""
    return (T_room + T_surface) / 2

def EstTemperatureOK(temps, T_room, T_surface):
    """Vérifie si la température est dans la plage de confort pendant les heures de bureau"""
    HeuresBureau = [8, 19]
    EN15251_temp = [19.5, 24]  # En °C, pas besoin de kelvin()
    
    if temps < HeuresBureau[0] or temps > HeuresBureau[1]:
        return False  # Hors des heures de bureau
    
    Temp_optimale = T_optimale(T_room, T_surface)
    return EN15251_temp[0] <= Temp_optimale <= EN15251_temp[1]

def Tmax_pour_deltaT(deltaT, T_dmax):
    """
    Fonction à annuler : T_max(deltaT) - T_dmax
    
    Arguments :
    - deltaT : durée de chauffage après 4h. (float64)
    - T_dmax : valeur cible de Tmax (ex : 24°C). (float64)
    
    Retourne :
    - Différence entre Tmax obtenu et Tmax souhaité.
    """

    t, T = calculTemperaturesEuler([0, 24], [15, 15, 15, 15, 15], 0.1, deltaT)
    T_confort = (T[0, :] + T[4, :]) / 2  # Troom = T[0], Tc2 = T[4]
    Tmax = np.max(T_confort)
    return Tmax - T_dmax

#______________________________________________________________________________________________________#
# question 4.2

#______________________________________________________________________________________________________#
# question 4.3

