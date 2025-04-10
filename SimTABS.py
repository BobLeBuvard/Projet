import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from PerteEtGain import g
from config import *

def dessinemoassa(t,T,index,xlabel = None, ylabel = None, titre= None):
    """
    Trace un ou plusieurs graphes des températures en fonction du temps.

    Paramètres :
    - t (ndarray, shape (1, ...)) : Instants de temps.
    - T (ndarray, shape (1, 5)) : Températures [T_room, T_t, T_cc, T_c1, T_c2].
    - index (list) : Titres des graphes correspondant aux courbes tracées.
    - Le reste est obvious.

    Retour :
    - Affiche le(s) graphique(s) demandé(s).
    """
    plt.xlabel(xlabel, fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.ylabel(ylabel, fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    for i in range(T.shape[0]):  
        plt.plot(t, T[i], label=index[i])  # En fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
    plt.legend( loc='best', fontsize = 7)
    plt.title(label = titre)
    plt.show()  

# SCENARIOS POUR LA QUESTION 3.4

def scenario1(t, delta_t = None):
    '''4h de refroidissement et puis le chauffage est coupé'''
    if 0<= t <=4 :
        heating_mode = 2 # refroidit
    else:
        heating_mode = 1 # éteint
    return heating_mode
def scenario2(t, delta_t = None):
    ''' 4h de refroidissement,10h de chauffe et puis le chauffage est coupé '''
    if 0<= t <=4 :
        heating_mode = 2 # refroidit
    elif 4<t<=13:
        heating_mode = 3 #chauffe
    else:
        heating_mode = 1 # éteint
    return heating_mode
def scenario3(t, delta_t = None): 
    '''12h de chauffe et puis 12h de refroidissement'''
    if 0<= t <=12 :
        heating_mode = 3 # chauffe
    else:
        heating_mode = 2 # refroidit
    return heating_mode
def scenario4(t, delta_t =None ):
    if delta_t == None: delta_t = 0 #Par défaut zéro...
    if 0<= t <=4 :
        heating_mode = 2 # refroidit
    elif 4<t<= (4+ delta_t):
        heating_mode = 3 # chauffe
    elif((4+delta_t)<t<=24 ):
        heating_mode = 1 # éteint
    else:
        # si t trop gand, osef de delta_t
        heating_mode = 1 # éteint
    return heating_mode

def scenario(t,num,delta_t = None): # delta_t = None définit s'il y a un argument supplémentaire (delta_t)
    '''
    on a défini 4 scénarios, cette fonction peut nous définir lequel on va utiliser pour notre fonction:
    - num (int) : numéro du scénario
    - t (float) : variable du scénario 
    - delta_t (float) : intervalle de temps ( utile que pour le scénario 4 )
    '''
    scenarios = [scenario1,scenario2,scenario3,scenario4]

    return scenarios[num-1](t%24,delta_t = delta_t) # %24 si les cycles sont de plus de 24h

def T_w(heating_mode,T_t):
    '''
    prend en entrée 1 , 2 ou 3
    - heating_mode == 1 : éteint (vaut T_t)
    - heating_mode == 2 : refroidit (vaut 18°C) 
    - heating_mode == 3 : chauffe (vaut 28°C)
    
    '''
    if heating_mode == 3:
        return 28
    elif heating_mode == 2: 
        return 18
    else:
        return T_t # le dernier terme est annulé donc il faut que T_t - T_w = 0 -> T_w = T_t


#______________________________________________________________________________________________________#
 #question 3.1
def odefunction(t, T,other_args):
    
    """
    Calcule les cinq dérivées des températures à l'instant t.

    Paramètres :
    - t (float64) : Instant de temps. (heures)
    - T (ndarray, shape (5,1)) : Températures [T_room, T_t, T_cc, T_c1, T_c2] (°C ou K, différences uniquement).

    Retourne :
    - dT (ndarray, shape (5,)) : Dérivées des températures à t (secondes)   .
    """


    delta_t = other_args.get('delta_t',None)
    num_du_scenario = other_args.get('num_du_scenario',1)
    dT = np.empty(len(T), dtype=np.float64) # de même dimensions que T mais contient les dérivées

    # CALCUL DE dT_room
    dT[0] = inv_C[0]*((-1/(R_r_moins_s + R_s_moins_c2))*(T[0]-T[4]) +g(t))
                    
    # CALCUL DE dT_t 

    heating_mode = scenario(t, num_du_scenario, delta_t)
    
    dT[1] = inv_C[1]*( (-1/R_x)*(T[1]-T[2]) - (1/R_w)*(T[1] - T_w(heating_mode, T[1])) )

    # CALCUL DE dT_cc
    dT[2] = inv_C[2]*( (-1/(R_cc_moins_c1))*(T[2]-T[3])- (1/R_x)*(T[2]-T[1]) + (1/R_c2_moins_cc)*(T[4] - T[2]))

    # CALCUL DE dT_c1 
    dT[3] = inv_C[3]*(-1/R_cc_moins_c1)*(T[3]-T[2])

    # CALCUL DE dT_c2 
    dT[4] = inv_C[4]* ((-1/R_c2_moins_cc)*(T[4]-T[2])+ (1/(R_r_moins_s + R_s_moins_c2))*(T[0] - T[4]))
    dT *= 3600
    return(dT)

#______________________________________________________________________________________________________#
#question 3.2 

def calculTemperaturesEuler(FenetreDeTemps, T0, h,**kwargs):
    """
    Résolution d'une équation différentielle par Euler.

    Paramètres :
    - FenetreDeTemps (ndarray, shape (2,)): Début et fin de la fenêtre de calcul (ex: [0, 24] pour 24h).
    - T0 (ndarray, shape (5,1)): Conditions initiales des températures.
    - h (int): Pas de temps pour la résolution.

    Retourne :
    - t (ndarray, shape (1, 24/h)): Instants de calcul.
    - T (ndarray, shape (5, 24/h)): Températures correspondantes.
    """
    
    # Initialisation des matrices
    n = int((FenetreDeTemps[1] - FenetreDeTemps[0]) / h) + 1   
    t = np.linspace(FenetreDeTemps[0], FenetreDeTemps[1], n)
    T = np.zeros((5, n))  
    T[:, 0] = T0  
    
    # Méthode de Euler
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1],kwargs)  #calcul des dérivées de tout pour chaque dernier élément de la colonne
        T[:, i] = T[:, i-1] + h * dT  
    return [t, T]

def question_3_2(**kwargs):

    # Initialisation des variables
    global gl_FenetreDeTemps,gl_h,gl_T0
    num_du_scenario = kwargs.get('num_du_scenario',gl_num_du_scenario) 
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',gl_FenetreDeTemps) 
    T0 = kwargs.pop('T0',gl_T0) 
    h = kwargs.pop('h',gl_h) 

    # Calcul
    t,T = calculTemperaturesEuler(FenetreDeTemps,T0,h,**kwargs)
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°C)',titre= f'Euler: scénario {num_du_scenario}, pas h = {h}')
    return [t,T]
#______________________________________________________________________________________________________#
#question 3.3

def calculTemperaturesIVP(FenetreDeTemps, T0, rtol,**kwargs):
    
    """
    Résout une équation différentielle par la méthode de Runge-Kutta (ode45).

    Paramètres :
    - FenetreDeTemps (ndarray, shape (2,)): Début et fin de la fenêtre de calcul (ex: [0, 24] pour 24h).
    - T0 (ndarray, shape (5,)): Conditions initiales des températures.
    - rtol (float): Tolérance de résolution (erreur maximale attendue).
    - t_eval (ndarray) : Points d'évaluation forcés pour comparer avec Euler.

    Retour :
    - Solution de l'ODE aux points spécifiés.
    """

    # Initialisation des variables
    t_eval = kwargs.get('t_eval',None)

    # Calcul
    solution = scp.integrate.solve_ivp(odefunction, FenetreDeTemps, T0, rtol= rtol,t_eval = t_eval,args=(kwargs,)) # forcer d'évaluer aux valeurs de t de Euler pour le dernier paramètre si on veut comparer Solve_IVP et Euler
    return[solution.t, solution.y]

def question_3_3(**kwargs):
    # Initialisation des variables
    global gl_FenetreDeTemps,gl_h,gl_T0
    num_du_scenario = kwargs.get('num_du_scenario',gl_num_du_scenario) 
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',gl_FenetreDeTemps) 
    T0 = kwargs.pop('T0',gl_T0) 
    kwargs.pop('h',gl_h) 
    IVP_tol = kwargs.pop('IVP_tol',gl_default_tol)
    # Calcul
    t,T = calculTemperaturesIVP(FenetreDeTemps,T0,IVP_tol,   **kwargs)

    # Dessin
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°C)',titre= f'IVP: scénario {num_du_scenario}')


#______________________________________________________________________________________________________#
#question 3.4

'''tester la différence entre les deux fonctions pour des valeurs de h différentes -> tester la convergence de Euler avec solve_IVP'''
def diff_entre_Euler_et_IVP():
    '''Fonction qui dessine des graphiques de la différence entre la résolution par Euler et par Runge-Kutta pour estimer leur convergence l'une vers l'autre'''
    # Initialisation des variables
    h_de_test = [0.001,0.01,0.1,0.25,0.5,1,2]

    # Calcul
    for i, h in enumerate(h_de_test):  #énumérer les éléments de h 
        h = h_de_test[i]
        t_euler,T2 = calculTemperaturesEuler(gl_FenetreDeTemps,gl_T0,h)
        _,T1 = calculTemperaturesIVP(gl_FenetreDeTemps,gl_T0, rtol=10e-10,t_eval=t_euler)
        T = T1 -T2
        dessinemoassa(t_euler,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'différence entre Euler et Runge avec h = {h_de_test[i]}')


def compare_avec_max(h_test,Max,**kwargs):
    
    # Initialisation des variables
    global gl_FenetreDeTemps,gl_T0
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',gl_FenetreDeTemps) 
    T0 = kwargs.pop('T0',gl_T0) 
    
    # Calcul des températures avec différentes précisions
    t_euler, T1 = calculTemperaturesEuler(FenetreDeTemps, T0, h_test,**kwargs)
    _, T2 = calculTemperaturesEuler(FenetreDeTemps, T0,Max ,**kwargs)
    ratio_tol = int(h_test / Max) 
    n_points = min(len(T1[0]), len(T2[0]) // ratio_tol) 
    difference_avec_max = np.zeros((T1.shape[0], n_points+1)) # n_points +1 car Euler compte le départ et l'arrivée
    
    # Comparaison des valeurs
    for i in range(n_points):  
        difference_avec_max[:, i+1] = T2[:, i * ratio_tol] - T1[:, i]

    # Dessin
    dessinemoassa(t_euler, difference_avec_max,['T_room','T_t','T_cc','T_c1','T_c2'])
    return difference_avec_max


def question_3_4():
    '''Fonction qui dessine des graphiques de la différence entre la résolution par Euler et par Runge-Kutta pour estimer leur convergence l'une vers l'autre'''
    diff_entre_Euler_et_IVP() 

#______________________________________________________________________________________________________#
#question 3.5

def afficher_scenario(t_total, T_total, FenetreDeTemps, num_du_scenario, delta_t, q_3_5, debug, i,isIVP):
    """Affiche le scénario en fonction des paramètres."""
    
    if q_3_5:
        for j in range(0, 5, 4):  # Prend les indices 0 et 4
            plt.plot(
                t_total / (FenetreDeTemps[1] - FenetreDeTemps[0]),
                T_total[j],
                label=['T_room', None, None, None, 'T_c2'][j]
            )
        if isIVP: isIVP = 'IVP' 
        else: isIVP = 'Euler'
        plt.title(f"T_room et T_c2 jusqu'à stagnation (sc.{num_du_scenario}) (delta_t = {round(delta_t, 2)}) ({isIVP})")
        plt.xlabel('nombre de cycles')
        plt.ylabel('températures des objets')
        plt.legend(loc='best')
        plt.show()

    elif not q_3_5 and debug:
        plt.plot(
            t_total / (FenetreDeTemps[1] - FenetreDeTemps[0]),
            (T_total[0] + T_total[4]) / 2,
            label = "température de confort"
        )
        plt.plot([0, i + 2], np.full(2, max(T_total[0] + T_total[4]) / 2),
                  linestyle = "--",
                  label = "température maximale"
                  )  # Trace la ligne du max
        plt.title(f"Température de confort jusqu'à stagnation (delta_t = {delta_t})")
        plt.xlabel('nombre de cycles')
        plt.ylabel('températures des objets')
        
        plt.legend(loc = 'best')
        plt.show()


def cycles_stab(T0, FenetreDeTemps,**kwargs):
    
    # Initialisation des variables
    global gl_h,tol_temp,max_jours,gl_num_du_scenario,gl_default_tol
    
    
    h = kwargs.pop('h',gl_h)
    
    num_du_scenario = kwargs.get('num_du_scenario',gl_num_du_scenario)
    delta_t = kwargs.get('delta_t',0)
    q_3_5 = kwargs.pop('q_3_5',True)
    max_jours = kwargs.pop('max_jours',max_jours)
    tol_temp = kwargs.pop('tol_temp',tol_temp)
    journee_pas = [FenetreDeTemps[0]]
    T_total = np.copy(T0).reshape(5,1)
    t_total = np.array([FenetreDeTemps[0]])
    isIVP = kwargs.pop('isIVP',False)
    IVP_tol = kwargs.pop('IVP_tol',gl_default_tol)
    # Calcul
    for i  in range(max_jours):
        
        if isIVP: 
            t,T = calculTemperaturesIVP(FenetreDeTemps, T_total[:,-1] ,IVP_tol,**kwargs)
        else: 
            t,T = calculTemperaturesEuler(FenetreDeTemps, T_total[:,-1] , h ,**kwargs)
        #ajoute à chaque fois ]0,24] si FenetreDeTemps = [0,24]
    
        T_total = np.concatenate((T_total,T[:,1:]),axis = 1)
        t_total = np.concatenate((t_total,t[1:]+(i)*(FenetreDeTemps[1]-FenetreDeTemps[0])))
        journee_pas.append(len(t))
        
        if abs(T[0, -1] - T_total[0,-(journee_pas[-1])]) < tol_temp:
            if debug: print(f"a stabilisé après {i+1} jours")

            # Dessin
            afficher_scenario(t_total, T_total, FenetreDeTemps, num_du_scenario, delta_t, q_3_5, debug, i,isIVP)
            return i+1 , T_total[:,-(1+journee_pas[-1])]  #retourne le nombre de jours et les conditions au début du dernier jour 
        if debug : print(f"n'a pas stabilisé après {i+1} jours")
    print(f"n'a pas convergé après {max_jours} jours, ajoutez plus de jours")
    return t_total,T_total

def question_3_5(**kwargs):
    # Initialisation des variables
    global gl_T0, gl_FenetreDeTemps
    T0 = kwargs.pop('T0',gl_T0)
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',gl_FenetreDeTemps)

    # Calcul
    return cycles_stab(T0, FenetreDeTemps,**kwargs)
    
#______________________________________________________________________________________________________#
# question 3.6
def question_3_6(**kwargs):
    """ Compare les différents scénarios. """
    # Initialisation des variables
    global gl_T0,gl_FenetreDeTemps
    T0,FenetreDeTemps = gl_T0,gl_FenetreDeTemps 
    kwargs.pop('num_du_scenario',None)

    # Calcul
    for i in range(3):
        cycles_stab(T0,FenetreDeTemps,     num_du_scenario = i+1,**kwargs)     