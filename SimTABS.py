import numpy as np
from main import odefunction
from main import kelvin
import scipy as scp
from config import *
import math

#question 3.2 
def calculTemperaturesEuler(FenetreDeTemps, T0, h ):
    t0, tf = FenetreDeTemps

    t = np.arange(t0, tf + h, h)  # on fait des temps discrets distancés de h entre t0 et tf 
    n = len(t)  # nombre de points de temps -> je préfère faire ainsi parce que on demande d'utiliser n , sinon je ferais T = np.zeros((5, len(t)))
    
    T = np.zeros((5, n))  # 5*n températures en fonction du nombre de points de temps -> on est obligé de mettre sous cette forme 
    T[:, 0] = T0  # conditions initiales
    
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1])  #calcul des dérivées
        T[:, i] = T[:, i-1] + h * dT  # application de Euler (copypaste du cours avec modifs)
        
    return [t, T]


#question 3.3
def calculTemperaturesIVP(FenetreDeTemps, T0, rtol, t_eval = None):
    solution = scp.integrate.solve_ivp(odefunction, FenetreDeTemps, T0, rtol= rtol,t_eval = t_eval) # forcer d'évaluer aux valeurs de t de Euler pour le dernier paramètreS
    return[solution.t, solution.y]




# question 4.1 base
def calculCycles(cycles,T0,FenetreDeTemps,h):
    '''

    fonction qui calcule un nombre de cycles de chauffe (sur plusieurs jours potentiellement) et qui retourne des données plottables. avec le calcul de températures par Euler
    
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

        if i > 0:
            t = t[:-1]
            T = T[:, :-1]

        t, T = calculTemperaturesEuler(FenetreDeTemps, T0, h ) #T0 est de dimensions (2,dimension de fenetre avec des increments de h) #PAR DEFAUT SI Custom = True, c'est sur 24h avec des incéments de 6min

        T_Total = np.concatenate((T_Total,T), axis = 1) 
        
        t_Total = np.concatenate((t_Total,(t + ((FenetreDeTemps[1]-FenetreDeTemps[0])*i) )))
        
        T0 = T[:, -1] #prendre les 5 dernières valeurs de l'itération précédentes comme valeurs initiales -> la dernière colonne de t et T


    return(t_Total,T_Total)

def converge(h, T_total,tolerance):
    ''' 
    Recherche si les températures finissent par stagner après un certain nombre de cycles
    
    h -> entier : intervalle entre les mesures. C'est 24h/h 
    
    T_total -> array[a,b]: array contenant les températures des 5 surfaces évaluées aux différents intervalles
    
    tolerance -> entier : tolérance à partir de laquelle on définit que la température stagne
    '''
    #conversion du temps donné en minutes -> 
    nextDay = round(24/h) #nombre de pas de temps dans 24h ATTENTION --> LE NOMBRE DE PAS DE TEMPS DOIT POUVOIR DIVISER 24H ( ex: des pas de temps de 0.9h ne peuvent pas faire des cycles complets de 24h)

    diff = np.zeros(T_total.shape[1] - nextDay) #liste vide de différences entre 2 jours --> on doit retirer l'équivalent d'un jour
   
    for i in range(T_total.shape[1]- nextDay) :
        diff[i] = abs(T_total[0][i] - T_total[0][i+ nextDay] ) 
        # print( str(i)+" : "  +str(diff[i])) #DEBUG
        if diff[i] <=  tolerance :
            print("a convergé après " +str((i/nextDay) +1) + "jours") # le +1 car il y a un décalage de 1 jour entre cycles et nombre de jours: après 10 jours il y a eu 9 cycles par exemple
            return diff 
    print("il n'y a pas eu convergence sur l'intervalle.")
    return diff
