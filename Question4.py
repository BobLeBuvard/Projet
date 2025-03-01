import matplotlib.pyplot as plt
from RechercheRacine import bissection
import numpy as np
from SimTabsFinal import calculTemperaturesEuler,kelvin,celsius,cycles_apres_convergence

def fonctionzero():
    plt.plot(np.arange(24),np.zeros(24))

#______________________________________________________________________________________________________#
# question 4.1


def T_max(delta_t, no_max = False, T0 = None):
    """
    Fonction qui calcule le maximum de température de confort d'un cycle (avec un delta T donné)
    
    Fonction à annuler : T_max(deltaT) - T_dmax 

    IN:

    - deltaT : durée de chauffage après 4h. (float64)

    - T_dmax : valeur cible de Tmax (ex : 24°C). (float64)
    
    OUT:
    
    - Différence entre Tmax obtenu et Tmax souhaité.
    """
    MAX = 0
    t, T = calculTemperaturesEuler([0, 24], T0,  0.01,num_du_scenario = 4, delta_t = delta_t)
    T_confort = (T[0, :] + T[4, :]) / 2  # T_room = T[0], T_c2 = T[4]
    if no_max == False:
        MAX = np.max(T_confort) #si on ne veut pas de max --> no_max=True ben on ne le calcule pas.
    return MAX,t,T_confort

def question_4_1(delta_t,T_max_d):
    MAX,t,T_confort = T_max(delta_t)
    print(celsius(MAX))
    plt.xlabel("temps (24h)", fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.ylabel("température optimale", fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.title(label = f'Température de confort sur 24h -> delta_t = {delta_t}')
    plt.plot(t,celsius(T_confort)-T_max_d ,label= "prout")
    fonctionzero()
    plt.legend( loc='best')
    plt.show()
 

#______________________________________________________________________________________________________#
#question 4.2

def recherche_delta_t (T_max_d, intervalle = [0,24], tol = 0.5e-7, T0 = kelvin(np.array([15, 15, 15, 15, 15]))):
    
    f_difference = lambda deltaT: T_max(deltaT, T0)[0] - T_max_d 
    '''
    fonction qui fait la différence entre T_max qui varie en fonction de delta et T_max_d qui est choisis abritrairement, il faut en 
    rechercher la racine pour pouvoir trouver delta_t
    '''
    delta_t ,statut = bissection(f_difference,intervalle[0],intervalle[1], tol=tol, max_iter=54)
    if statut !=0 :
        print('erreur')
        return(-1)
    return delta_t


#______________________________________________________________________________________________________#
# question 4.3



#EN15251 est une array contenant t0 et tf et Tmin et Tmax -> [8,19,19.5,24]
EN15251 = np.array([8,19,19.5,24])
def verification_EN15251(delta_t,EN15251):
    MAX,t,T_max_arr = T_max(delta_t,no_max = True)
    T_confort = T_max_arr
    
    for i in range(len(t)):
        T_confort_i = T_confort[i]
        if (EN15251[0]<= t[i]<=  EN15251[1]):
            if not(EN15251[2] < T_confort_i < EN15251[3]): 
                print("La norme EN15251 n'est pas respectée.")
                return False
    print("La norme EN15251 est respectée.")
    return True

def question_4_3(T_max_d,EN15251, T0 = kelvin(np.array([15, 15, 15, 15, 15]))):
    h = 0.01
    
    FenetreDeTemps = [0,24]
    days_to_converge, T0_new = cycles_apres_convergence(T0,FenetreDeTemps,h) #T0_new est les conditions initiales du dernier jour
    if days_to_converge == None:
        print("Les températures ne se sont pas stabilisées.")
        return -1
    delta_t = recherche_delta_t(T_max_d,T0 = T0_new)
    return verification_EN15251(delta_t,EN15251)