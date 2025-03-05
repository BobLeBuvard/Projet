import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from PerteEtGain import g
from config import *

def kelvin(temp):
    return (temp+273.15) 
def celsius(temp):
    return (temp-273.15)
def dessinemoassa(t,T,index,xlabel = None, ylabel = None, titre= None):
    ''' fonction qui plotte le graphe de ce qu'on lui a donné.
     IN: 
    
     t -> array d'instant de temps (float64), dim(1,...)
    
     T -> array des températures (dim (1,5)) dans l'ordre [T_room, T_t, T_cc, T_c1,T_c2]

     index -> un array de titre de graphe (comme la fonction peut en dessiner plusieurs d'un coup)

     xlabel -> nom de l'axe des abcisses, par défaut = None, il n'y en a pas

     ylabel -> nom de l'axe des ordonnés, par défaut = None, il n'y en a pas

     titre -> nom du graphique, par défaut = None, il n'y en a pas

     OUT:

     Graphique représentant les différentes choses demandées en paramètre
    '''

    plt.xlabel(xlabel, fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.ylabel(ylabel, fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    for i in range(T.shape[0]):  
        plt.plot(t, T[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
    plt.legend( loc='best')
    plt.title(label = titre)
    plt.show()  

#SCENARIOS POUR LA QUESTION 4
# delta_t = None c'est pour singaler qu'il peut y avoir des arguments supplémentaires dans la fonction. Dans notre cas, delta_t

def scenario1(t, delta_t = None):
    '''4h de refroidissement et puis le chauffage est coupé'''
    if 0<= t <=4 :
        heating_mode = 2 #refroidit
    else:
        heating_mode = 1 #éteint
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
        heating_mode = 3 #chauffe
    else:
        heating_mode = 2 #refroidit
    return heating_mode
def scenario4(t, delta_t =None ):
    if delta_t == None: delta_t = 0 #Par défaut zéro...
    if 0<= t <=4 :
        heating_mode = 2 # refroidit
    elif 4<t<= (4+ delta_t):
        heating_mode = 3 #chauffe
    elif((4+delta_t)<t<=24 ):
        heating_mode = 1 # éteint
    return heating_mode

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
            


def scenario(t,num,delta_t = None): # delta_t = None définit s'il y a un argument supplémentaire (delta_t)
    '''
    on a défini 4 scénarios, cette fonction peut nous définir lequel on va utiliser pour notre fonction:
    
    num -> numéro du scénario
    
    t -> variable du scénario 

    delta_t  -> intervalle de temps ( utile que pour le scénario 4 )
    '''
    scenarios = [scenario1,scenario2,scenario3,scenario4,scenario5]

    return scenarios[num-1](t,delta_t = delta_t)

def T_w(heating_mode,T_t):
    '''
    prend en entrée 1 , 2 ou 3

    heating_mode == 1 -> éteint (vaut T_t)

    heating_mode == 2 -> refroidit (vaut 18°C)
    
    heating_mode == 3 -> en mode chauffe (vaut 28°C)
    
    '''
    if heating_mode == 3:
        return kelvin(28)
    elif heating_mode == 2: 
        return kelvin(18)
    else:
        return T_t #le dernier terme est annulé donc il faut que T_t - T_w = 0 -> T_w = T_t


#______________________________________________________________________________________________________#
                                         #question 3.1
def odefunction(t, T,num_du_scenario = 1, delta_t = None,Force_heating = False):
    '''retourne une array contenant les cinq dérivées selon leur formule
    
    IN: 
    
    t -> instant de temps (float64)
    
    T -> array des températures (dim (1,5)) dans l'ordre [T_room, T_t, T_cc, T_c1,T_c2]

    delta_t = intervalle de temps (float64) supplémentaire pour le scénario 4, par défaut n'est pas utilisé (défini) = None

    num_du_scenario -> scenario choisi, de 1 à 5 (5 = customisé), par défaut le 1 est utilisé

    Force_heating -> entier de 1 à 3 . 1 éteint, 2 refroidit et 3 chauffe. Par défaut on ne force pas le chauffage (ce pourquoi False)

    OUT:

    dT -> dérivées des températures à l'instant t (dim(5))

    '''
    dT = np.zeros_like(T) # de même dimensions que T mais contient les dérivées

    #CALCUL DE dT_room
    dT[0] = (1/C[0])*((-1/(R_r_moins_s + R_s_moins_c2))*(T[0]-T[4]) +g(t))
                    
    #CALCUL DE dT_t 

    heating_mode = scenario(t, num_du_scenario, delta_t=delta_t)
    if Force_heating != False: heating_mode = Force_heating
    dT[1] = (1/C[1])*( (-1/R_x)*(T[1]-T[2]) - (1/R_w)*(T[1] - T_w(heating_mode, T[1])) )

    #CALCUL DE dT_cc
    dT[2] = (1/C[2])*( (-1/(R_cc_moins_c1))*(T[2]-T[3])- (1/R_x)*(T[2]-T[1]) + (1/R_c2_moins_cc)*(T[4] - T[2]))

    #CALCUL DE dT_c1 
    dT[3] = (1/C[3])*(-1/R_cc_moins_c1)*(T[3]-T[2])

    #CALCUL DE dT_c2 
    dT[4] = (1/C[4])* ((-1/R_c2_moins_cc)*(T[4]-T[2])+ (1/(R_r_moins_s + R_s_moins_c2))*(T[0] - T[4]))

    return(dT*3600)


#______________________________________________________________________________________________________#
#question 3.2 

def calculTemperaturesEuler(FenetreDeTemps, T0, h,num_du_scenario = 1, delta_t = None,Force_heating = False):
    '''
    Fonction qui résoud une équation différentielle par la méthode d'Euler:

    IN:

    FenetreDeTemps -> array de 2 éléments: le début (0) et la fin de la fenêtre de temps de calcul (généralement 24 pour 24h) de dimensions 1

    T0 -> conditions initiales (arrray de dimensions (5,1) ) 
    
    h -> pas de temps nécessaire à la résolution avec Euler (entier)
   
    delta_t -> temps (float64) supplémentaire pour le scénario 4, par défaut n'est pas utilisé (défini) = None

    num_du_scenario -> scenario choisi, de 1 à 5 (5 = customisé), par défaut le 1 est utilisé

    Force_heating -> entier de 1 à 3 . 1 éteint, 2 refroidit et 3 chauffe. Par défaut on ne force pas le chauffage (ce pourquoi False)

    OUT:
    
    t -> un array des temps, dim(1,24/h)

    T -> un array des température correspondantes aux temps utilsiés, dim(5,24/h)

    '''
    T0 = kelvin(T0)
    t0, tf = FenetreDeTemps

    t = np.arange(t0, tf + h, h)  # on fait des temps discrets distancés de h entre t0 et tf 
    n = len(t)  # nombre de points de temps -> je préfère faire ainsi parce que on demande d'utiliser n , sinon je ferais T = np.zeros((5, len(t)))
    
    T = np.zeros((5, n))  # 5*n températures en fonction du nombre de points de temps -> on est obligé de mettre sous cette forme 
    T[:, 0] = T0  # conditions initiales
    
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1], num_du_scenario, delta_t=delta_t,Force_heating = Force_heating)  #calcul des dérivées de tout pour chaque dernier élément de la colonne
        T[:, i] = T[:, i-1] + h * dT  # application de Euler 
    return [t, celsius(T)]

def question_3_2(num_du_scenario = 1):
    t,T = calculTemperaturesEuler(FenetreDeTemps,T0,h,num_du_scenario=num_du_scenario)
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'Euler: scénario {num_du_scenario}')

#______________________________________________________________________________________________________#
#question 3.3

def calculTemperaturesIVP(FenetreDeTemps, T0, rtol,num_du_scenario = 1, t_eval = None,delta_t = 0, Force_heating = False):
    '''
    Fonction qui résoud une équation différentielle par la méthode de Runge-Kutta (ode45):

    IN:

    FenetreDeTemps -> array de 2 éléments: le début (0) et la fin de la fenêtre de temps de calcul (généralement 24 pour 24h) de (array dim(1))

    T0 -> conditions initiales (array dim(5,0) )
    
    rtol -> tolérance de résolution ( à quoi on doit s'attendre comme différence avec la véritable valeur)

    t_eval -> paramètre pour forcer l'évaluation aux points de Euler pour pouvoir comparer à des t identiques. On pourrait interpoler mais je sais pas trop
    '''
    T0 = kelvin(T0)
    solution = scp.integrate.solve_ivp(odefunction, FenetreDeTemps, T0, rtol= rtol,t_eval = t_eval,args=(num_du_scenario, delta_t, Force_heating)) # forcer d'évaluer aux valeurs de t de Euler pour le dernier paramètre si on veut comparer Solve_IVP et Euler
    return[solution.t, celsius(solution.y)]

def question_3_3(num_du_scenario = 1):
    t,T = calculTemperaturesIVP(FenetreDeTemps,T0,10e-10,num_du_scenario = num_du_scenario)
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'IVP: scénario {num_du_scenario}')


#______________________________________________________________________________________________________#
#question 3.4

'''tester la différence entre les deux fonctions pour des valeurs de h différentes -> tester la convergence de Euler avec solve_IVP'''
def diff_entre_Euler_et_IVP():
    '''Fonction qui dessine des graphiques de la différence entre la résolution par Euler et par Runge-Kutta pour estimer leur convergence l'une vers l'autre'''

    h_de_test = [0.001,0.01,0.1,0.25,0.5,1,2]
    for i in range(len(h_de_test)):
        
        h = h_de_test[i]
        t_euler,T2 = calculTemperaturesEuler(FenetreDeTemps,T0,h)
        t,T1 = calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10,t_eval=t_euler)
        
        
        T = T1 -T2
        if debug: print(f'T1 est de dimensions: {T1.shape} et T2 est de dimensions: {T2.shape}')
        dessinemoassa(t_euler,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'différence entre Euler et Runge avec h = {h_de_test[i]}')
def question_3_4():
    '''Fonction qui dessine des graphiques de la différence entre la résolution par Euler et par Runge-Kutta pour estimer leur convergence l'une vers l'autre'''
    diff_entre_Euler_et_IVP() 

def compare_avec_max(h_test):
    '''fonction test pour comparer avec le maximum de précision --> a supprimer, c'est juste pour le choix de h'''
    difference_avec_max = []
    t_euler,T1 = calculTemperaturesEuler(FenetreDeTemps,T0,h_test)
    t_max,T2 = calculTemperaturesEuler(FenetreDeTemps,T0,10e-6)
    ratio_tol = h_test/10e-6
    for i in range(len(T1)):
        difference_avec_max.append(T2[i]-T1[i*ratio_tol]) 
    plt.plot(t_euler,difference_avec_max)
    plt.show
    return difference_avec_max


#______________________________________________________________________________________________________#
#question 3.5

def cycles_apres_convergence(T0, FenetreDeTemps, h,delta_t = 0,num_du_scenario = 1, tol=0.01, max_jours=30, Force_heating = False, q_3_5=True):
    '''
    fonction qui va calculer itérativement chaque jour et va voir à partir de quand la température se stabilise entre les jours.

    IN: 

    T0 -> conditions initiales en celsius(array dim(5,1))

    FenetreDeTemps -> durée d'un cycle

    h -> pas du calcul de la fonction (0.01)

    tol -> tolérance de convergence (par défault = 0.01), erreur maximum tolérée pour dire que les températures convergent

    max_jour -> Si ce nombre de jours est dépassé, la fonction s'arrête même si les températures n'ont pas convergé après x jours (30)
    par défaut)

    delta_t -> intervalle de temps de chauffe utile pour la question 4
    
    Force_heating -> entier de 1 à 3 . 1 éteint, 2 refroidit et 3 chauffe. Par défaut on ne force pas le chauffage (ce pourquoi False)
    
    

    OUT :

    Graphique des température sur la durée qui assure la convergence.


    '''
    # calculer les 2 premiers jours
    journee_pas = round((FenetreDeTemps[1]-FenetreDeTemps[0])/h)
    t,T = calculCycles(2,T0,FenetreDeTemps,h,delta_t = delta_t)
    T_total = np.copy(T)
    t_total = np.copy(t)
    for i  in range(max_jours-2):
        if abs(T_total[0, -1] - T_total[0, -(1+journee_pas)]) <= tol:
            print(f"a convergé après {i+2} jours")
            if q_3_5:
                for j in range(0,5,4):
                    plt.plot(t_total/(FenetreDeTemps[1]-FenetreDeTemps[0]),T_total[j])
                plt.title(label = f"graphique de la température jusqu'à convergence (delta_t = {delta_t})")
                plt.xlabel('nombre de cycles')
                plt.ylabel('températures des objets')
                plt.show()
            if not q_3_5:
                plt.plot(t_total/(FenetreDeTemps[1]-FenetreDeTemps[0]),(T_total[0]+T_total[4])/2)
                plt.title(label = f"graphique de la température de confort jusqu'à convergence (delta_t = {delta_t})")
                plt.xlabel('nombre de cycles')
                plt.ylabel('températures des objets')
                plt.plot([0,i+2],np.full(2,max(T_total[0]+T_total[4])/2)) #plotter le maximum
                plt.show()
            return i+2 , T_total[:,-(1+journee_pas)]  #retourne le nombre de jours et les conditions au début du dernier jour
            
        else:
            t,T = calculTemperaturesEuler(FenetreDeTemps, T_total[:,-1] , h ,num_du_scenario= num_du_scenario,delta_t = delta_t,Force_heating= Force_heating)
            #ajouter le dernier jour à T_total et t_total
            T_total = np.concatenate((T_total,T),axis = 1)
            t_total = np.concatenate((t_total,t+(i+2)*(FenetreDeTemps[1]-FenetreDeTemps[0])))
            print(f"n'a pas convergé après {i+2} jours")
    print(f"n'a pas convergé après {max_jours} , ajoutez plus de jours")
    return None, None


def calculCycles(cycles,T0,FenetreDeTemps,h,delta_t = 0,Force_heating = False):
    '''

    Fonction qui calcule un nombre de cycles de chauffe (sur plusieurs jours potentiellement) et qui retourne des données plottables. avec le calcul de températures par Euler
    
    IN: 

    cycles: nombre de cycles d'évaluation (int)

    T0-> températures initiales sous forme d'array de dimensions(1,5) avec les éléments [T_room, T_t, T_cc, T_c1,T_c2]
    
    FenetreDeTemps: durée d'un cycle sous forme d'array/liste [t0,tf] (ex: [0,24] -> cycle de 24h)
    
    h-> intervalle entre les instants de calcul de température (float64)

    OUT: 
    
    t->  temps d'évaluation (array de array dim(1) de longueur -> intervalle/h )
    
    T-> array de dimensions (5, cycles*h +1)
        '''

    T_Total = np.empty((5, 0))  # 5 lignes, 0 colonnes
    t_Total = np.array([])
    for i in range(cycles):
        if i > 0:
            t = t[:-1]
            T = T[:, :-1]
        t, T = calculTemperaturesEuler(FenetreDeTemps, T0, h,delta_t = delta_t, Force_heating =Force_heating) #T0 est de dimensions [5,0]
        T_Total = np.concatenate((T_Total,T), axis = 1) 
        
        t_Total = np.concatenate((t_Total,(t + ((FenetreDeTemps[1]-FenetreDeTemps[0])*i) )))
        
        T0 = T[:, -1] #prendre les 5 dernières valeurs de l'itération précédentes comme valeurs initiales -> la dernière colonne de t et T


    return(t_Total,T_Total)

def dessineDesCycles(cycles,num_du_scenario):
    t,T = calculCycles(cycles,T0,FenetreDeTemps,0.01)
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'Euler: scénario {num_du_scenario}')

def question_3_5():
    '''fonction qui va dessiner le graphe tes températures d'une journée jusqu'à arriver à un état staionnaire'''
    cycles_apres_convergence(T0, FenetreDeTemps,0.01)



#______________________________________________________________________________________________________#
# question 3.6
def question_3_6():
    for i in range(3):
        t,T = calculTemperaturesEuler(FenetreDeTemps,T0,h,num_du_scenario = i+1)
        dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'scénario {i+1}')
