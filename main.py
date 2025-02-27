#DEPRECATED: NE PLUS UTILISER LES FONCTIONS QUI SONT ICI SAUF POUR DES TESTS

import numpy as np
from config import * 
import SimTABS
from Question4 import EstTemperatureOK
from SimTabsFinal import calculTemperaturesEuler

def convergeEfficace(h, T0, tolerance, temp=0):
    """
    Fonction qui vérifie la convergence des températures après plusieurs cycles.

    h : intervalle entre les mesures
    T0 : température initiale (array de taille (5,))
    tolerance : tolérance définissant la convergence
    temp : index de la température à analyser (par défaut 0)
    --> POSE DES SOUCIS ! CA MARCHE PAS
    """

    T_Total = np.empty((5, 0))  # Initialisation de la matrice vide
    t_Total = np.array([])  # Initialisation du temps vide
    j = 0  # compteur de cycles
    nextDay = round(24 / h)  # Nombre de pas de temps dans 24h

    # Calcul des 2 premiers jours
    t, T = SimTABS.calculCycles(2, T0, FenetreDeTemps, h)

    # Stocker les premières valeurs
    T_Total = np.copy(T)
    t_Total = np.arange(len(t))  # Stocker le temps

    # Liste pour stocker les températures du dernier instant de chaque journée
    last_temps = [T[temp][-1]]  

    while j < 28:  # Maximum 30 jours
        # Vérification de la convergence avec le dernier instant de chaque jour
        if len(last_temps) > 1 and abs(last_temps[-1] - last_temps[-2]) <= tolerance:
            print(f"Convergence après {j+2} jours")
            print(abs(last_temps[-1] - last_temps[-2]))
            print(last_temps[-2])
            print(last_temps[-1])
            return [T_Total,T, last_temps]

        j += 1
        print(f"Jour {j+2}: pas encore de convergence")

        # Calcul du cycle suivant
        t, T_add = SimTABS.calculCycles(1, T[:, -1], FenetreDeTemps, h)

        # Mise à jour des matrices
        T_Total = np.concatenate((T_Total, T_add), axis=1)
        t_Total = np.concatenate((t_Total, t + 24 * j))

        # Ajout de la nouvelle température du dernier instant du jour
        last_temps.append(T_add[temp][-1])

    print("Erreur : la convergence n'a pas été atteinte en 30 jours.")
    return [T_Total, last_temps, "erreur: convergence de plus de 30 jours"]

def converge(h, T_total,tolerance):
    ''' 
    Recherche si les températures finissent par stagner après un certain nombre de cycles
    
    h -> entier : intervalle entre les mesures. C'est 24h/h 
    
    T_total -> array[a,b]: array contenant les températures des 5 surfaces évaluées aux différents intervalles
    
    tolerance -> entier : tolérance à partir de laquelle on définit que la température stagne
    '''
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

def T_confort_max(FenetreDeTemps, T0, h):
    while(delta_t <24):
        delta_t += 0.5
        t, T = calculTemperaturesEuler(FenetreDeTemps, T0, h )
        for i in range(t): #tester pour tous les éléments de T
            if not EstTemperatureOK(i,T[0],T[4]): 
                break