#DEPRECATED: NE PLUS UTILISER LES FONCTIONS QUI SONT ICI SAUF POUR DES TESTS DE RETROCOMPATIBILITE

import numpy as np
from config import * 
import SimTABS_old
import math
from Question4 import T_max
from RechercheRacine import bissection,secante

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
    t, T = SimTABS_old.calculCycles(2, T0, FenetreDeTemps, h)

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
        t, T_add = SimTABS_old.calculCycles(1, T[:, -1], FenetreDeTemps, h)

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

def converge_fin_journee(T_total, tolerance,h):
    """
    Vérifie la convergence de la température à la fin de chaque journée.


    T_total : matrice des températures au fil du temps (arrayarray dim(5, n))
    
    tolerance : seuil de tolérance pour considérer une convergence (float64)

    h : intervalle entre les mesures (en heures) (float64)

    Retourne : le tableau des différences entre jours successifs (dim(5,n))

    """

    StepsInADay = round(24 / h) + 1  # Nombre de points dans une journée
    num_days = (T_total.shape[1] - 1) // StepsInADay  # Nombre total de jours utilisables

    if num_days < 2:  # On doit comparer au moins 2 jours
        print("Pas assez de jours pour tester la convergence.")
        return np.array([])

    diff = np.zeros(num_days - 1)

    for i in range(num_days - 1):
        diff[i] = abs(T_total[0, (i + 1) * StepsInADay] - T_total[0, i * StepsInADay])

        if diff[i] <= tolerance:
            print(f"a convergé après {i+2} jours")
            return diff, i+2  # On arrête dès qu'on a une convergence

    print("il n'y a pas eu convergence sur l'intervalle.")
    return diff

def bisecante(f, x0, x1, tol = 0.5e-7, max_iter=50):
    for i in range(max_iter):
        pass

        #ICI ON DOIT METTRE LA METHODE DE LA BISSECTION POUR RETRECIRE L'INTERVALLE
        if i > math.ceil((max_iter)/4): 
            break


    #passage à la méthode de la sécante 

        
    fx1 = f(x1)     #STOCKAGE DE L'ESTIMATION DE LA FONCTION A UN POINT X1
    fx0 = f(x0)     #STOCKAGE DE L'ESTIMATION DE LA FONCTION A UN POINT X0

    if abs(fx1 - fx0) ==0 : # on veut pas diviser par zéro
        return [1984, -1]
    #maintenant on calcule la formule du point de la fonction à l:

    x2 = x1 -  fx1* (x1 - x0) / (fx1 - fx0) # on calcule le nouveau point x
    if abs(x2 - x1) < tol:
        return [x2, 0] #on a trouvé la racine correcte (avec tolérances en valeur absolue) !

    x0, x1 = x1, x2  # Sinon on met les valeurs à jour x1 devient x0 et x2 devient x1  
    return [x2,0]

#methode de la sécante
def Recherchede_Delta_d(Td_max, intervalle, T0, h, G_interp):
    """Trouve la durée Δt pour atteindre Tmax = Td_max sur 24h en utilisant la méthode de la sécante."""
    def erreur_delta(dt):
        return T_max(intervalle, T0, h, G_interp) - Td_max
    
    dt_opt, statut = secante(erreur_delta, 1, 10, 1e-2)
    if statut == 0:
        return dt_opt
    else:
        print("Échec de la recherche de Δt")
        return None
    
def kelvin(temp):
    return (temp+273.15) 
def celsius(temp):
    return (temp-273.15)


def scenario5(t,delta_t = None):
    '''
    En fonction d'une variable on peut créer un scénario de chauffe customisé

    scénario à sa guise: on entre une array pour dire les heures durant lesquelles on veut chauffer, refroidir ou couper
    
    Cette fonction prend delta_t comme paramètre en compte, mais delta_t dans ce cas n'est pas une simple array

    dans le scénario 5 en effet, delta_t est un tuple contenant un nombre: le mode de chauffe de base ( 1, 2 ou 3) et une array d'heures de chauffe
     l'array d'heures de chauffe est une array plate 3xn  -> Par exemple si on veut faire chauffer entre 5 et 6h on met [3(type de chauffe),5(début),6(fin)]    
    
    Voici un exemple de delta_t

            

    delta_t = (1,np.array([ 3,5,6, 2,9,10, 3,15,18, 3,8,9])
               ^            ^ ^ ^
               |            | | |
    (défaut éteint)   chaud-début-fin

    '''

    if type(delta_t) is not int:
        default_mode,heatingcycle = delta_t
    else:
        print("il y a un problème de delta_t: delta_t n'est pas celui attendu (tuple) contenant un int et une flat_array")
        heating_mode = False
    if heatingcycle!= None:
        matrice = heatingcycle.reshape(3, len(heatingcycle/3))
        '''
        exemple de matrice
    mode    3  2  2  3 
    début   5  9  15 8 
    fin     6  10 18 9

        '''
        for i in range(np.shape[1]): # nombre de colonnes
            if matrice[1,i] <= t <= matrice[2,i]:
                heating_mode = matrice[0,i]
                return heating_mode
    #on a pas trouvé de valeur pour laquelle on veut chauffer ou refroidir.
    return default_mode

    if debug:
        plt.plot(t_total,T_total[0])
        plt.show()


def dessineDesCycles(cycles,**kwargs):
    global gl_num_du_scenario,gl_T0,gl_FenetreDeTemps
    num_du_scenario = kwargs.get('num_du_scenario',gl_num_du_scenario)
    t,T = calculCycles(cycles,gl_T0,gl_FenetreDeTemps,**kwargs)
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'Euler: scénario {num_du_scenario}')


#DEPRECATED: NE PLUS UTILISER LES FONCTIONS QUI SONT ICI SAUF POUR DES TESTS

import numpy as np
from SimTABS import odefunction
from Question4 import *
import scipy as scp
from config import *

#question 3.2 
def calculTemperaturesEuler(FenetreDeTemps, T0, h ):
    t0, tf = FenetreDeTemps

    t = np.arange(t0, tf + h, h)  # on fait des temps discrets distancés de h entre t0 et tf 
    n = len(t)  # nombre de points de temps -> je préfère faire ainsi parce que on demande d'utiliser n , sinon je ferais T = np.zeros((5, len(t)))

    T = np.zeros((5, n))  # 5*n températures en fonction du nombre de points de temps -> on est obligé de mettre sous cette forme 
    T[:, 0] = T0  # conditions initiales

    for i in range(1, n):
    dT = odefunction(t[i-1], T[:, i-1])  #calcul des dérivées de tout pour chaque dernier élément de la colonne
    T[:, i] = T[:, i-1] + h * dT  # application de Euler 
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

def converge_fin_journee(T_total, tolerance,h):
    """
    Vérifie la convergence de la température à la fin de chaque journée.

    h : intervalle entre les mesures (en heures)
    T_total : matrice des températures au fil du temps (shape = (5, n))
    tolerance : seuil de tolérance pour considérer une convergence

    Retourne : le tableau des différences entre jours successifs
    """

    StepsInADay = round(24 / h) + 1  # Nombre de points dans une journée
    num_days = (T_total.shape[1] - 1) // StepsInADay  # Nombre total de jours utilisables

    if num_days < 2:  # On doit comparer au moins 2 jours
        print("Pas assez de jours pour tester la convergence.")
        return np.array([])

    diff = np.zeros(num_days - 1)

    for i in range(num_days - 1):
        diff[i] = abs(T_total[0, (i + 1) * StepsInADay] - T_total[0, i * StepsInADay])

        if diff[i] <= tolerance:
            print(f"a convergé après {i+2} jours")
            return diff  # On arrête dès qu'on a une convergence

    print("il n'y a pas eu convergence sur l'intervalle.")
    return diff

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
    t, T = calculCycles(2, T0, FenetreDeTemps, h)

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
        t, T_add = calculCycles(1, T[:, -1], FenetreDeTemps, h)

        # Mise à jour des matrices
        T_Total = np.concatenate((T_Total, T_add), axis=1)
        t_Total = np.concatenate((t_Total, t + 24 * j))

        # Ajout de la nouvelle température du dernier instant du jour
        last_temps.append(T_add[temp][-1])

    print("Erreur : la convergence n'a pas été atteinte en 30 jours.")
    return [T_Total, last_temps, "erreur: convergence de plus de 30 jours"]

#______________________________________________________________________________________________________#
#question 3.4 mauvais
def question_4_3(T_max_d, EN15251 = np.array([8,19,19.5,24]), T0 = kelvin(np.array([15, 15, 15, 15, 15]))):

    h = 0.01
    FenetreDeTemps = [0,24]
    delta_t = recherche_delta_t(kelvin(T_max_d),T0 = T0) #OK
    print(f"on a trouvé un delta_t de {delta_t}")
    question_4_1(delta_t, T_max_d)   #montrer le graphe du premier jour 
    days_to_converge, T0_new = cycles_apres_convergence(T0,FenetreDeTemps,h,delta_t= delta_t,num_du_scenario=4 ) #T0_new est les conditions initiales du dernier jour
    if days_to_converge == None:
        print("Les températures ne se sont pas stabilisées.")
        return -1
    plt.show()

    return verification_EN15251(delta_t,EN15251,T0_new)

def testRacine():
    import time
    sc = [bissection, secante, hybride]
    x0 = 0
    x1 = 10
    def f(x):
        time.sleep(0.1)
        return x**2 -3
    for i in range(3):
        start = time.time()
        racine, statut = sc[i](f,x0,x1)
        print(f"racine et statut: {racine,statut}")
        end = time.time()
        print(f"temps {end-start}")

#DEPRECATED: NE PLUS UTILISER LES FONCTIONS QUI SONT ICI SAUF POUR DES TESTS

from config import *
import SimTABS_old
import matplotlib.pyplot as plt
# sim (simulation) c'est l'exercice de test qu'on veut appliquer.
# 1 c'est les cycles
#2 c'est la différence (! au custom = True et euler qui fait des siennes )
#3 c'est calcul par euler
#4 par IVP

def test(h):

    global T1,T2
    t_euler,T2 = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h)
    t,T1 = SimTABS_old.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10,t_eval=t_euler )


    T3 = T1 -T2
    if debug: 
        plt.ylabel('Température(T)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
        plt.xlabel('Temps (t)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
        index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
        for i in range(T3.shape[0]):  
            plt.plot(t, T3[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
        plt.legend( loc='best')
        plt.title(label = 'h = '+str(h))
        plt.show()  


sim= 2

#TESTER LES EXERCICES

if sim == 1:
t,T = SimTABS_old.calculCycles(5,T0,FenetreDeTemps,h)

#question 3.4
elif sim ==2:
'''tester la différence entre les deux fonctions pour des valeurs de h différentes -> tester la convergence de Euler avec solve_IVP'''
h_de_test = [0.001, 0.01,0.1,0.25,0.5,1,24]
for i in range(len(h_de_test)):
    test(h_de_test[i])        

elif sim == 3:
'''calcul de température par Euler'''
t,T = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h)



elif sim == 4: 
'''Calcul par solve_IVP'''
h = 'méthode de solve_IVP'
t,T = SimTABS_old.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10)


elif sim ==5:
h = 0.01
''' calcul de la différence de température par tranches de 24h --> NECESSITE UN h DIVISEUR DE 24 ( ex: 0.1, 0.01, ou autre )'''
t,T = SimTABS_old.calculCycles(16,T0,FenetreDeTemps,h)
T_converge = SimTABS_old.converge(h,T,0.01)
t2 = t[:len(T_converge)] # les  premiers éléments de t2
plt.plot (t2,T_converge)
plt.title(label = 'graphique de la différence de température entre deux jours au cours du temps')
plt.plot()
plt.show() 
#main.dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'],ylabel='Température(T)',xlabel ='Temps (t)' ,titre = str(h))

elif sim == 6:
'''tester la différence entre les résolutions par Euler pour des valeurs de h différentes '''
h_de_test = [0.001, 0.01,0.1,0.25] #0.001 pas utile puisque 0.01 l'approche suffisemment bien
for i in range(len(h_de_test) -1 ):
    t_euler2,T2 = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h_de_test[i])   
    t_euler1,T1 = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h_de_test[i+1])
    for i in range(5):
        plt.plot(t_euler1,T1[i])
        plt.plot(t_euler2,T2[i])
        plt.show()




if debug and (sim !=2) and (sim !=6)  : 
#T = T - 273.15 #remise en celsius
plt.ylabel('Température(°K)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
plt.xlabel('Temps (heures)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
for i in range(T.shape[0]):  
    plt.plot(t, T[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
plt.legend( loc='best')
plt.title(label = str(h))
plt.show()  


def fonctiondroite(hauteur, label = None):
    '''fonction qui va plot y = 0 sur le graphique'''
    global gl_FenetreDeTemps
    plt.plot(np.arange(gl_FenetreDeTemps[1]-gl_FenetreDeTemps[0]+1),np.zeros(gl_FenetreDeTemps[1]-gl_FenetreDeTemps[0]+1) + hauteur , label = label)
