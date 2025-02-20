import numpy as np
from scipy.integrate import solve_ivp as solve
from PerteEtGain import g



#FORME DE l'array T 

# T = np.array([T_room, T_t, T_cc, T_c1,T_c2])

num_du_scenario = 1

C_room = 12 # Capacité de la pièce régulée (kJ/m²K)
C_c1 = 50 # Capacité de la partie supérieure béton
C_c2 = 10 # Capacité de la partie inférieure béton
C_cc = 50 # Capacité de la partie centrale béton
C_t = 1 # Capacité des tubes
C_w = 30 # Capacité de l'eau
C = np.array([C_room, C_t, C_cc, C_c1,C_c2, C_w])

R_s = 1 #Résistance de la surface entrre le béton et la pièce régulée
R_x = 0.025 #Résistance de contact entre les tubes & la partie centrale du béton (m²K/W)
R_w = 0.15 #Résistance de l'eau
R_cc_moins_c1 = 0.025
R_c2_moins_cc = 0.02
R_r_moins_s = 0.1
R_s_moins_c2 = 0.183




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


def scenario(num,t):
    '''on a défini 3 scénarios, cette fonction peut nous définir lequel on va utiliser pour notre fonction:
    
    num -> numéro du scénario
    
    t -> variable du scénario 
    '''
    scenarios = [scenario1,scenario2,scenario3]

    return scenarios[num-1](t)


def T_w(isOn,T_t):
    '''
    prend en entrée 1 , 2 ou 3

    isOn == 1 -> éteint (vaut T_t)

    isOn == 2 -> refroidit (vaut 18°C)
    
    isOn == 3 -> en mode chauffe (vaut 28°C)
    
    '''
    if isOn == 3:
        return 28
    elif isOn == 2: 
        return 18
    else:
        return T_t #le dernier terme est annulé donc il faut que T_t - T_w = 0 -> T_w = T_t

#question 3.1
def odefunction(t, T):
    
    '''retourne une array contenant les cinq dérivées selon leur formule'''


    dT = np.zeros(5)


    #CALCUL DE dT_room
    dT[0] = (1/C[0])*((-1/(R_r_moins_s +R_s_moins_c2))*(T[0]-T[4]+g(t)))  #Il manque la fonction G(t)
                    

    #CALCUL DE dT_t 
    dT[1] = (1/C[5])*( (-1/R_x)*(T[1]-T[2]) - (1/R_w)*(T[1] - T_w(scenario(num_du_scenario,t), T[1])) )


    #CALCUL DE dT_cc
    dT[2] = (1/C[2])*( (-1/(R_cc_moins_c1))*(T[2]-T[3])- (1/R_x)*(T[2]-T[1]) + (1/R_c2_moins_cc)*(T[4] - T[2]))

    #CALCUL DE dT_c1 
    dT[3] = (1/C[3])*(-1/R_cc_moins_c1)*(T[3]-T[2])

    #CALCUL DE dT_c2 
    dT[4] = (1/C[4])* ((-1/R_c2_moins_cc)*(T[4]-T[2])+ (1/(R_r_moins_s + R_s_moins_c2))*(T[0] - T[4]))

    return(dT)


# inutile pour le moment
# def T_optimale(T_room, T_surface):
#     '''calcule la température ressentie en fonction de la chaleur de la pièce et celles des surfaces'''
#     return((T_room+T_surface)/2)



