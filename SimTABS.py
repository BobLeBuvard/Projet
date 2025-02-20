import numpy as np
from main import odefunction
import scipy as scp

#question 3.2 
def calculTemperaturesEuler(FenetreDeTemperature, T0, h):
    t0, tf = FenetreDeTemperature

    t = np.arange(t0, tf + h, h)  # on fait des temps discrets distancés de h entre t0 et tf
    n = len(t)  # nombre de points de temps -> je préfère faire ainsi parce que on demande d'utiliser n , sinon je ferais T = np.zeros((5, len(t)))
    
    T = np.zeros((5, n))  # 5*n températures en fonction du nombre de points de temps -> on est obligé de mettre sous cette forme 
    T[:, 0] = T0  # conditions initiales
    
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1])  #calcul des dérivées
        T[:, i] = T[:, i-1] + h * dT  # application de Euler
        
    return [t, T]


#question 3.3
def calculTemperaturesIVP(FenetreDeTemperature, T0, rtol = 10^-7): # par défaut une précision de 10^-7 (choix arbitraire)
    solution = scp.integrate.solve_ivp(odefunction, FenetreDeTemperature, T0, rtol)
    return(solution)
# question 4.1 base
def calculCycles(cycles,T0,FenetreDeTemps,h):
    '''

    fonction qui calcule un nombre de cycles de chauffe (sur plusieurs jours potentiellement) et qui retourne des données plottables.
    
    ======
    IN: 

    cycles: nombre de cycles d'évaluation 

    T0: températures initiales sous forme d'array de dimensions(1,5) avec les éléments [T_room, T_t, T_cc, T_c1,T_c2]
    
    FenetreDeTemps: durée d'un cycle sous forme d'array [t0,tf] (ex: [0,24] -> cycle de 24h)
    
    h: intervalle entre les instants de calcul de température

    ======
    OUT: 
    

    t: temps d'évaluation
    
    T: array de dimensions (5, cycles*h + cycles-1 ( souci de compter 2 fois la fin d'un cycle et le début d'un cycle suivant) ) -> pour 1 cycle avec h = 24 c'est (5,24+1)
    '''
    T_Total = np.empty((5, 0))  # 5 lignes, 0 colonnes
    t_Total = np.array([])

    for i in range(cycles):
        t, T = calculTemperaturesEuler(FenetreDeTemps, T0, h) #T0 est de dimensions (2,dimension de fenetre avec des increments de h)

        T_Total = np.concatenate((T_Total,T), axis = 1) 
        
        t_Total = np.concatenate((t_Total,(t + ((FenetreDeTemps[1]-FenetreDeTemps[0])*i) )))
        
        T0 = [T[0][-1],T[1][-1],T[2][-1],T[3][-1],T[4][-1]] #prendre les 5 dernières valeurs de l'itération précédentes comme valeurs initiales -> le dernier élément de la rangée correspondante

    T = T_Total #flemme de changer tous les "plot", je renomme juste T 
    t = t_Total #idem qu'au dessus
    return(t,T)
