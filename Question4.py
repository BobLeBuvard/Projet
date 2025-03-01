from SimTABS import calculTemperaturesEuler
import math
import matplotlib.pyplot as plt
from RechercheRacine import bissection,secante
import numpy as np
from SimTabsFinal import calculTemperaturesEuler,kelvin,celsius,converge_fin_journee, calculCycles

def fonctionzero():
    plt.plot(np.arange(24),np.zeros(24))

#______________________________________________________________________________________________________#
# question 4.1


def T_max(delta_t, no_max = False):
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
    t, T = calculTemperaturesEuler([0, 24], kelvin(np.array([15, 15, 15, 15, 15])),  0.01,num_du_scenario = 4, delta_t = delta_t)
    T_confort = (T[0, :] + T[4, :]) / 2  # T_room = T[0], T_c2 = T[4]

    if not no_max: MAX = np.max(T_confort) #si on ne veut pas de max --> no_max=True ben on ne le calcule pas.
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

def recherche_delta_t (T_max_d, intervalle = [0,24], tol = 0.5e-7):
    
    f_difference = lambda deltaT: T_max(deltaT)[0] - T_max_d 
    '''''
    fonction qui fait la différence entre T_max qui varie en fonction de delta et T_max_d qui est choisis abritrairement, il faut en 
    rechercher la racine pour pouvoir trouver delta_t
    '''''
    delta_t ,statut = bissection(f_difference,intervalle[0],intervalle[1], tol=tol, max_iter=54)
    if statut !=0 :
        print('erreur')
        return(-1)
    return delta_t
'''

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
'''

#______________________________________________________________________________________________________#
# question 4.3



#EN15251 est une array contenant t0 et tf et Tmin et Tmax -> [8,19,19.5,24]

def verification_EN15251(delta_t,EN15251):
    MAX,t,T_max_arr = T_max(delta_t,no_max = True)
    T_confort = T_max_arr
    for i in range(len(t)):
        if EN15251[0]<= t[i]<=  EN15251[1] and not EN15251[2] < T_confort[i] < EN15251[3]: 
            print("La norme EN15251 n'est pas respectée.")
            return False
    print("La norme EN15251 est respectée.")
    return True

def question_4_3(T_max_d):
    t,T = calculCycles(10,kelvin(np.array([15,15,15,15,15])), [0,24],0.01)
    convergenceArray,has_converged = converge_fin_journee(T,0.01,0.01)
    if has_converged == None:
        print("l'intervalle de jours de test est trop petit pour avoir le temps de converger, calculer plus de cycles")
        return(-1)
    
    delta_t = recherche_delta_t(T_max_d)
    verifie_il = verification_EN15251(delta_t)
    return(verifie_il)