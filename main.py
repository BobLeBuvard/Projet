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
