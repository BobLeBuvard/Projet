import numpy as np
from main import odefunction

def calculTemperaturesEuler(interval, T0, h):
    t0, tf = interval
    t = np.arange(t0, tf + h, h)  # Discrétisation du temps
    n = len(t)  # Nombre de points de temps
    
    T = np.zeros((5, n))  # Stockage des températures pour chaque instant
    T[:, 0] = T0  # Conditions initiales
    
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1])  # Calcul des dérivées
        T[:, i] = T[:, i-1] + h * dT  # Mise à jour des températures avec Euler
        