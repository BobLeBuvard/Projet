import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from PerteEtGain import g


def kelvin(temp):
    return (temp+273.15) 
def celsius(temp):
    return (temp-273.15)

#CONFIG
FenetreDeTemps = np.array([0, 24]) # fenetre de test comme demandé -> taille du cycle
h = 0.01  # pas de temps ( toutes les 6 minutes)
T0 = kelvin(np.array([15,15,15,15,15])) #conitions initiales données -> ici mises en array en kelvins
nombre_de_cycles = 11
default_tol = 10e-10 #choix arbitraire

#FORME DE l'array T 

# T = np.array([T_room, T_t, T_cc, T_c1,T_c2])

C_room = 12 # Capacité de la pièce régulée (kJ/m²K)
C_c1 = 50# Capacité de la partie supérieure béton (kJ/m²K)
C_c2 = 10# Capacité de la partie inférieure béton (kJ/m²K)
C_cc = 50 # Capacité de la partie centrale béton (kJ/m²K)
C_w = 30 # Capacité de l'eau (kJ/m²K)
C = np.array([C_room, C_w, C_cc, C_c1,C_c2, ])*1000 # kJ/m²K -> J/m²K


R_x = 0.025 #Résistance de contact entre les tubes & la partie centrale du béton (m²K/W)
R_w = 0.15 #Résistance de l'eau
R_cc_moins_c1 = 0.05
R_c2_moins_cc = 0.02
R_r_moins_s = 0.1
R_s_moins_c2 = 0.183

debug = True

def dessinemoassa(t,T,index,xlabel = None, ylabel = None, titre= None ):
    ''' fonction qui plotte le graphe de ce qu'on lui a donné.
     IN: 
    
     t -> array d'instant de temps (float64), dim(1,...)
    
     T -> array des températures (dim (1,5)) dans l'ordre [T_room, T_t, T_cc, T_c1,T_c2]

     index -> un array de titre de graphe (comme la fonction peut en dessiner plusieurs d'un coup)

     xlabel -> nom de l'axe des abcisses, par défault = none, il n'y en a pas

     ylabel -> nom de l'axe des ordonnés, par défault = none, il n'y en a pas

     titre -> nom du graphique, par défault = none, il n'y en a pas

     OUT:
     Graphique déssiné
    '''

    #TODO: expliciter in et out avec le même format (à vérif)
    if debug:
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
    if delta_t.type != 'int':
        default_mode,heatingcycle = delta_t
    else:
        print("il y a un problème de delta_t: delta_t n'est pas celui attendu (tuple) contenant un int et une flat_array")
        heating_mode = False
    if heatingcycle!= None:
        matrice = heatingcycle.reshape(3, heatingcycle.shape/3)
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
    #TODO: mettre à jour le docstring avec les paramètres (maj faite, à vérif)
    '''retourne une array contenant les cinq dérivées selon leur formule
    
    IN: 
    
    t -> instant de temps (float64)
    
    T -> array des températures (dim (1,5)) dans l'ordre [T_room, T_t, T_cc, T_c1,T_c2]

    delta_t = ,intervalle de temps (float64) supplémentaire pour le scénario 4, par défault n'est pas utilisé (défini) = none

    num_descenario -> scenario choisi, de 1 à 5 (5 = debug), par défaut = 1

    Force_heating -> bool (flase or true), default = false, si on met ce paramètre, on peut dire si on veut chauffer, refroidire ou couper sur le cycle complet. (on peut aussi le faire avec le scénario 5 et delta_t = )


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
    #TODO: mettre à jour le docstring avec les paramètres (à vérif)
    '''
    Fonction qui résoud une équation différentielle par la méthode d'Euler:

    IN:

    FenetreDeTemps -> array de 2 éléments: le début (0) et la fin de la fenêtre de temps de calcul (généralement 24 pour 24h) de dimensions 1

    T0 -> conditions initiales (arrray de dimensions (5,1) ) 
    
    h -> pas de temps nécessaire à la résolution avec Euler (entier)
   
    delta_t -> temps (float64) supplémentaire pour le scénario 4, par défaut n'est pas utilisé (défini) = none

    num_descenario -> scenario choisi, de 1 à 5 (5 = debug), par défaut le 1 est utilisé

    Force_heating -> bool (flase or true), default = false, si on met ce paramètre, on peut dire si on veut chauffer, refroidire ou couper sur le cycle complet. (on peut aussi le faire avec le scénario 5 et delta_t = )

    OUT:
    
    t -> un array des temps, dim(1,24/h)

    T -> un array des température correspondantes aux temps utilsiés, dim(5,24/h)

    '''
    t0, tf = FenetreDeTemps

    t = np.arange(t0, tf + h, h)  # on fait des temps discrets distancés de h entre t0 et tf 
    n = len(t)  # nombre de points de temps -> je préfère faire ainsi parce que on demande d'utiliser n , sinon je ferais T = np.zeros((5, len(t)))
    
    T = np.zeros((5, n))  # 5*n températures en fonction du nombre de points de temps -> on est obligé de mettre sous cette forme 
    T[:, 0] = T0  # conditions initiales
    
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1], num_du_scenario, delta_t=delta_t,Force_heating = Force_heating)  #calcul des dérivées de tout pour chaque dernier élément de la colonne
        T[:, i] = T[:, i-1] + h * dT  # application de Euler 
    return [t, T]
def question_3_2(num_du_scenario = 1):
    t,T = calculCycles(1,T0,FenetreDeTemps,0.01)
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'Euler: scénario {num_du_scenario}')

#______________________________________________________________________________________________________#
#question 3.3

def calculTemperaturesIVP(FenetreDeTemps, T0, rtol, t_eval = None):
    '''
    Fonction qui résoud une équation différentielle par la méthode de Runge-Kutta (ode45):

    IN:

    FenetreDeTemps -> array de 2 éléments: le début (0) et la fin de la fenêtre de temps de calcul (généralement 24 pour 24h) de (array dim(1))

    T0 -> conditions initiales (array dim(5,0) )
    
    rtol -> tolérance de résolution ( à quoi on doit s'attendre comme différence avec la véritable valeur)

    t_eval -> paramètre pour forcer l'évaluation aux points de Euler pour pouvoir comparer à des t identiques. On pourrait interpoler mais je sais pas trop
    '''
    solution = scp.integrate.solve_ivp(odefunction, FenetreDeTemps, T0, rtol= rtol,t_eval = t_eval) # forcer d'évaluer aux valeurs de t de Euler pour le dernier paramètre si on veut comparer Solve_IVP et Euler
    return[solution.t, solution.y]

def question_3_3(num_du_scenario = 1):
    t,T = calculTemperaturesIVP(FenetreDeTemps,T0,10e-10)
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


#______________________________________________________________________________________________________#
#question 3.5

def cycles_apres_convergence(T0, FenetreDeTemps, h, tol=0.01, max_jours=30):
    '''
    fonction qui va calculer itérativement chaque jour et va voir à partir de quand la température se stabilise entre les jours.
    
    IN: 

    T0 -> conditions initiales (array dim(5,1))

    FenetreDeTemps -> durée d'un cycle

    '''
    # calculer les 2 premiers jours
    journee_pas = round((FenetreDeTemps[1]-FenetreDeTemps[0])/h)
    t,T = calculCycles(2,T0,FenetreDeTemps,h)
    T_total = np.copy(T)
    t_total = np.copy(t)
    for i  in range(max_jours):
        if abs(T_total[0, -1] - T_total[0, -(1+journee_pas)]) <= tol:
            print(f"a convergé après {i+2} jours")
            
            if debug:
                plt.plot(t_total,T_total)
                plt.title(label= "graphique de la température jusqu'à convergence")
                plt.show()
            return i+2 , T_total[:,-(1+journee_pas)]  #retourne le nombre de jours et les conditions de l'avant dernier jour
            
        else:
            t,T = calculTemperaturesEuler(FenetreDeTemps,T_total[:,-1],h)
            #ajouter le dernier jour à T_total et t_total
            T_total = np.concatenate((T_total,T),axis = 1)
            t_total = np.concatenate((t_total,t))
            print(f"n'a pas convergé après {i+2} jours")
    print(f"n'a pas convergé après {max_jours} , ajoutez plus de jours")
    return None, None


def calculCycles(cycles,T0,FenetreDeTemps,h):
    '''

    Fonction qui calcule un nombre de cycles de chauffe (sur plusieurs jours potentiellement) et qui retourne des données plottables. avec le calcul de températures par Euler
    
    IN: 

    cycles: nombre de cycles d'évaluation (int)

    T0: températures initiales sous forme d'array de dimensions(1,5) avec les éléments [T_room, T_t, T_cc, T_c1,T_c2]
    
    FenetreDeTemps: durée d'un cycle sous forme d'array [t0,tf] (ex: [0,24] -> cycle de 24h)
    
    h: intervalle entre les instants de calcul de température (float64)

    OUT: 
    

    t: temps d'évaluation (array de array dim(1) de longueur -> intervalle/h )
    
    T: array de dimensions (5, cycles*h + cycles-1 ( souci de compter 2 fois la fin d'un cycle et le début d'un cycle suivant) ) -> pour 1 cycle avec h = 24 c'est (5,24+1)
    ex: dim(5, 24000)
        '''

    T_Total = np.empty((5, 0))  # 5 lignes, 0 colonnes
    t_Total = np.array([])
    for i in range(cycles):
        if i > 0:
            t = t[:-1]
            T = T[:, :-1]
        t, T = calculTemperaturesEuler(FenetreDeTemps, T0, h ) #T0 est de dimensions [5,0]
        T_Total = np.concatenate((T_Total,T), axis = 1) 
        
        t_Total = np.concatenate((t_Total,(t + ((FenetreDeTemps[1]-FenetreDeTemps[0])*i) )))
        
        T0 = T[:, -1] #prendre les 5 dernières valeurs de l'itération précédentes comme valeurs initiales -> la dernière colonne de t et T


    return(t_Total,T_Total)

def dessineDesCycles(cycles,num_du_scenario):
    t,T = calculCycles(cycles,T0,FenetreDeTemps,0.01)
    dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'Euler: scénario {num_du_scenario}')

def question_3_5():
    '''fonction qui va dessiner le graphe tes températures d'une journée jusqu'à arriver à un état staionnaire'''
    jours,T_2 = cycles_apres_convergence(T0, FenetreDeTemps,0.01)
    plt.plot(np.arange(len(T_2))*h,T_2)



#______________________________________________________________________________________________________#
# question 3.6
def question_3_6():
    for i in range(3):
        t,T = calculTemperaturesEuler(FenetreDeTemps,T0,h,num_du_scenario = i+1)
        dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'scénario {i}')
