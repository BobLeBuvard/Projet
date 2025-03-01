import math
import matplotlib.pyplot as plt
from RechercheRacine import bissection
import numpy as np
from SimTabsFinal import calculTemperaturesEuler,kelvin
#______________________________________________________________________________________________________#
# question 4.1



def Trouver_T_max(delta_t, T_d_max):

    """
    Fonction qui calcule le maximum de température de confort d'un cycle (avec un delta T donné)
    
    Fonction à annuler : T_max(deltaT) - T_dmax 

    IN:

    - deltaT : durée de chauffage après 4h. (float64)

    - T_dmax : valeur cible de Tmax (ex : 24°C). (float64)
    
    OUT:
    
    - Différence entre Tmax obtenu et Tmax souhaité.
    """
    
    t, T = calculTemperaturesEuler([0, 24], kelvin(np.array([15, 15, 15, 15, 15])),  0.01,num_du_scenario = 4, delta_t = delta_t)
    T_confort = (T[0, :] + T[4, :]) / 2  # T_room = T[0], T_c2 = T[4]
    T_max = np.max(T_confort)
    return (T_max ,t,T_confort)

def question_4_1(delta_t):
    MAX,t,T_confort = Trouver_T_max(delta_t,kelvin(24.5))
    plt.xlabel("température optimale", fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.ylabel("temps (24h)", fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.title(label = f'Température de confort sur 24h -> delta_t = {delta_t}')
    plt.plot(t,T_confort,label= "prout")
    plt.legend( loc='best')
    plt.show()  
    

#______________________________________________________________________________________________________#
#question 4.2

def recherche_delta_t(T_max_d,x0,xf,tol = 0.5e-7):
    
    f_difference = lambda deltaT: Trouver_T_max(deltaT) - T_max_d 
    delta_t  = bissection(f_difference, 0 , 24 , 0.5e-7, max_iter=54)
    return delta_t

'''''

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

'''

#______________________________________________________________________________________________________#
# question 4.3
