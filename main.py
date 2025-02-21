import numpy as np
from PerteEtGain import g
from config import * 




def scenariodebug(t,delta_t):
    isOn = 1 #éteint pour voir la température de la pièce sans chauffe
    return isOn 

def scenario1(t,delta_t):
    '''4h de refroidissement et puis le chauffage est coupé'''
    if 0<= t <=4 :
        isOn = 2 #refroidit
    else:
        isOn = 1 #éteint
    return isOn
def scenario2(t,delta_t):
    ''' 4h de refroidissement,10h de chauffe et puis le chauffage est coupé '''
    if 0<= t <=4 :
        isOn = 2 # refroidit
    elif 4<t<=13:
        isOn = 3 #chauffe
    else:
        isOn = 1 # éteint
    return isOn
def scenario3(t,delta_t):
    '''12h de chauffe et puis 12h de refroidissement'''
    if 0<= t <=12 :
        isOn = 3 #chauffe
    else:
        isOn = 2 #refroidit
    return isOn

def scenario4(t,delta_t):
    if 0<= t <=4 :
        isOn = 2 # refroidit
    elif 4<t<= (4+delta_t):
        isOn = 3 #chauffe
    elif((4+delta_t)<t<=24 ):
        isOn = 1 # éteint
    return isOn

def scenario(t,num,delta_t):
    '''on a défini 4 scénarios, cette fonction peut nous définir lequel on va utiliser pour notre fonction:
    
    num -> numéro du scénario
    
    t -> variable du scénario 

    delta_t  -> intervalle de temps ( utile que pour le scénario 4 )
    '''
    scenarios = [scenario1,scenario2,scenario3,scenario4,scenariodebug]

    return scenarios[num-1](t,delta_t)


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

    '''retourne une array contenant les cinq dérivées selon leur formule
    
    IN: 
    
    t -> instant de temps (float64)
    
    T -> array des températures (dim (1,5)) dans l'ordre [T_room, T_t, T_cc, T_c1,T_c2]

    '''


    dT = np.zeros_like(T) # de même dimensions que T mais contient les dérivées


    #CALCUL DE dT_room
    dT[0] = (1/C[0])*((-1/(R_r_moins_s +R_s_moins_c2))*(T[0]-T[4]+g(t)))  #Il manque la fonction G(t)
                    

    #CALCUL DE dT_t 
    
    isOn = scenario( t ,num_du_scenario, delta_t = 0)
    # if debug: print(num_du_scenario) #DEBUG
    dT[1] = (1/C[5])*( (-1/R_x)*(T[1]-T[2]) - (1/R_w)*(T[1] - T_w(isOn, T[1])) )


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



