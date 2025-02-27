import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from PerteEtGain import g


def kelvin(temp):
    return (temp+273.15) 
def celcius(temp):
    return (temp-273.15)

#CONFIG
FenetreDeTemps = np.array([0, 24]) # fenetre de test comme demandé -> taille du cycle
num_du_scenario = 1
h = 0.01  # pas de temps ( toutes les 6 minutes)
debug = True
T0 = kelvin(np.array([15,15,15,15,15])) #conitions initiales données -> ici mises en array en kelvins
delta_t = 0
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

def dessinemoassa(t,T,index,xlabel = None, ylabel = None, titre= None ):
    ''' fonction qui plotte le graphe de ce qu'on lui a donné.'''
    if debug : 
        plt.ylabel(ylabel, fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
        plt.xlabel(xlabel, fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
        for i in range(T.shape[0]):  
            plt.plot(t, T[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
        plt.legend( loc='best')
        plt.title(label = titre)
        plt.show()  

#SCENARIOS POUR LA QUESTION 4
def scenario1(t):
    '''4h de refroidissement et puis le chauffage est coupé'''
    if 0<= t <=4 :
        isOn = 2 #refroidit
    else:
        isOn = 1 #éteint
    return isOn
def scenario2(t):
    ''' 4h de refroidissement,10h de chauffe et puis le chauffage est coupé '''
    if 0<= t <=4 :
        isOn = 2 # refroidit
    elif 4<t<=13:
        isOn = 3 #chauffe
    else:
        isOn = 1 # éteint
    return isOn
def scenario3(t):
    '''12h de chauffe et puis 12h de refroidissement'''
    if 0<= t <=12 :
        isOn = 3 #chauffe
    else:
        isOn = 2 #refroidit
    return isOn
def scenario4(t):
    if 0<= t <=4 :
        isOn = 2 # refroidit
    elif 4<t<= (4+delta_t):
        isOn = 3 #chauffe
    elif((4+delta_t)<t<=24 ):
        isOn = 1 # éteint
    return isOn
def scenario(t,num):
    '''on a défini 4 scénarios, cette fonction peut nous définir lequel on va utiliser pour notre fonction:
    
    num -> numéro du scénario
    
    t -> variable du scénario 

    delta_t  -> intervalle de temps ( utile que pour le scénario 4 )
    '''
    scenarios = [scenario1,scenario2,scenario3,scenario4]

    return scenarios[num-1](t)

def T_w(isOn,T_t):
    '''
    prend en entrée 1 , 2 ou 3

    isOn == 1 -> éteint (vaut T_t)

    isOn == 2 -> refroidit (vaut 18°C)
    
    isOn == 3 -> en mode chauffe (vaut 28°C)
    
    '''
    if isOn == 3:
        return kelvin(28)
    elif isOn == 2: 
        return kelvin(18)
    else:
        return T_t #le dernier terme est annulé donc il faut que T_t - T_w = 0 -> T_w = T_t


#______________________________________________________________________________________________________#
                                         #question 3.1
def odefunction(t, T):
    
    '''retourne une array contenant les cinq dérivées selon leur formule
    
    IN: 
    
    t -> instant de temps (float64)
    
    T -> array des températures (dim (1,5)) dans l'ordre [T_room, T_t, T_cc, T_c1,T_c2]

    OUT:
    dT -> dérivées des températures à l'instant t (dim(5))

    '''

    dT = np.zeros_like(T) # de même dimensions que T mais contient les dérivées

    #CALCUL DE dT_room
    dT[0] = (1/C[0])*((-1/(R_r_moins_s + R_s_moins_c2))*(T[0]-T[4]) +g(t))
                    
    #CALCUL DE dT_t 

    isOn = scenario( t ,num_du_scenario)
    dT[1] = (1/C[1])*( (-1/R_x)*(T[1]-T[2]) - (1/R_w)*(T[1] - T_w(isOn, T[1])) )

    #CALCUL DE dT_cc
    dT[2] = (1/C[2])*( (-1/(R_cc_moins_c1))*(T[2]-T[3])- (1/R_x)*(T[2]-T[1]) + (1/R_c2_moins_cc)*(T[4] - T[2]))

    #CALCUL DE dT_c1 
    dT[3] = (1/C[3])*(-1/R_cc_moins_c1)*(T[3]-T[2])

    #CALCUL DE dT_c2 
    dT[4] = (1/C[4])* ((-1/R_c2_moins_cc)*(T[4]-T[2])+ (1/(R_r_moins_s + R_s_moins_c2))*(T[0] - T[4]))

    return(dT*3600)


#______________________________________________________________________________________________________#
#question 3.2 

def calculTemperaturesEuler(FenetreDeTemps, T0, h ):
    '''
    Fonction qui résoud une équation différentielle par la méthode d'Euler:

    IN:

    FenetreDeTemps -> array de 2 éléments: le début (0) et la fin de la fenêtre de temps de calcul (généralement 24 pour 24h) de dimensions 1

    T0 -> conditions initiales (arrray de dimensions (5,0) ) 
    
    h -> pas de temps nécessaire à la résolution avec Euler (entier)
    '''
    t0, tf = FenetreDeTemps

    t = np.arange(t0, tf + h, h)  # on fait des temps discrets distancés de h entre t0 et tf 
    n = len(t)  # nombre de points de temps -> je préfère faire ainsi parce que on demande d'utiliser n , sinon je ferais T = np.zeros((5, len(t)))
    
    T = np.zeros((5, n))  # 5*n températures en fonction du nombre de points de temps -> on est obligé de mettre sous cette forme 
    T[:, 0] = T0  # conditions initiales
    
    for i in range(1, n):
        dT = odefunction(t[i-1], T[:, i-1])  #calcul des dérivées de tout pour chaque dernier élément de la colonne
        T[:, i] = T[:, i-1] + h * dT  # application de Euler 
    return [t, T]
def question_3_2():
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

    paramètre t_eval -> paramètre pour forcer l'évaluation aux points de Euler pour pouvoir comparer à des t identiques. On pourrait interpoler mais je sais pas trop
    '''
    solution = scp.integrate.solve_ivp(odefunction, FenetreDeTemps, T0, rtol= rtol,t_eval = t_eval) # forcer d'évaluer aux valeurs de t de Euler pour le dernier paramètre si on veut comparer Solve_IVP et Euler
    return[solution.t, solution.y]
def question_3_3():
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
        print(f'T1 est de dimensions: {T1.shape} et T2 est de dimensions: {T2.shape}')
        if debug: 
            dessinemoassa(t_euler,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'différence entre Euler et Runge avec h = {h_de_test[i]}')
def question_3_4():
    diff_entre_Euler_et_IVP() 


#______________________________________________________________________________________________________#
#question 3.5

def calculCycles(cycles,T0,FenetreDeTemps,h):
    '''

    Fonction qui calcule un nombre de cycles de chauffe (sur plusieurs jours potentiellement) et qui retourne des données plottables. avec le calcul de températures par Euler
    
    ======
    IN: 

    cycles: nombre de cycles d'évaluation (int)

    T0: températures initiales sous forme d'array de dimensions(1,5) avec les éléments [T_room, T_t, T_cc, T_c1,T_c2]
    
    FenetreDeTemps: durée d'un cycle sous forme d'array [t0,tf] (ex: [0,24] -> cycle de 24h)
    
    h: intervalle entre les instants de calcul de température (float64)

    ======
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
            return diff  # On arrête dès qu'on a une convergence

    print("il n'y a pas eu convergence sur l'intervalle.")
    return diff
def dessineDesCycles(cycles):
    if debug:
        t,T = calculCycles(cycles,T0,FenetreDeTemps,0.01)
        dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'Euler: scénario {num_du_scenario}')

def question_3_5():
    t,T = calculCycles(nombre_de_cycles,T0,FenetreDeTemps,h)
    T_2 = converge_fin_journee(T,0.01,h)
    plt.plot(np.arange(len(T_2)),T_2)



#______________________________________________________________________________________________________#
# question 3.6
def question_3_6():
    global num_du_scenario
    for i in range(3):
        num_du_scenario = (i+1)
        t,T = calculTemperaturesEuler(FenetreDeTemps,T0,h)
        dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='Temps (heures)',ylabel='Température(°K)',titre= f'scénario {i}')
