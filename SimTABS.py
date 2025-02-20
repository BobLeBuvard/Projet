import numpy as np
from main import odefunction

def calculTemperaturesEuler(interval, T0, h):
    t0, tf = interval
    t = np.arange(t0, tf + h, h)  # on fait des temps discrets distancés de h entre t0 et tf
    n = len(t)  # nombre de points de temps -> je préfère faire ainsi parce que on demande d'utiliser n , sinon je ferais T = np.zeros((5, len(t)))
    
    T = np.zeros((5, n))  # 5*n températures en fonction du nombre de points de temps -> on est obligé de mettre sous cette forme 
    T[:, 0] = T0  # conditions initiales
    
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1])  #calcul des dérivées
        T[:, i] = T[:, i-1] + h * dT  # application de Euler
        
    return [t, T]


T0 = [15, 15, 15, 15, 15]
interval = np.array([0, 24])  # En heures
h = 0.1  # Pas de temps

t, T = calculTemperaturesEuler(interval, T0, h)
print(T)        
'''bizarrement dans ce test après 24h, la température globale est aux alentours de 150°C ce qui est un pôtipeu chaud'''