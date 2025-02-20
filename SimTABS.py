import numpy as np
import scipy as scp
from main import odefunction
#DEBUG VARIABLES T: A RETIRER DANS LE CODE FINAL
T_room = 293,15 #kelvins = 20 degrés Celcius
T_c1 = 293,15 #kelvins 
T_c2 = 293,15 #kelvins
T_cc = 293,15 #kelvins
T_t = 293,15 #Température des tubes 
T0 = np.array([T_room, T_t, T_cc, T_c1,T_c2])
FenetreDeTemperature = np.array([0,24])
n = 5 #HARDCODED POUR DEBUG

#question 3.2
def calculTemperatureEuler(FenetreDeTemperature, T0, h):
    t0 = FenetreDeTemperature[0]
    tf = FenetreDeTemperature[1]
    t = np.zeros(5) #retour debug
    T = np.zeros(5*n) #retour debug
    temps = np.linspace(t0, tf, h)
    y[:, 0] = T0
    fun = lambda temps, y: odefunction(t, y, c)

    for i in range(1, 1001):
    y[:, i] = y[:, i-1] + 0.1 * fun(t[i-1], y[:, i-1]) # Méthode d'Euler

    return [t,T]

#question 3.3
def calculTemperaturesIVP(FenetreDeTemperature, T0, rtol = 10^-10):
    
    return(t,T)