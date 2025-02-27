from SimTABS import calculTemperaturesEuler
import math
from RechercheRacine import hasRoots
import numpy as np
from SimTabsFinal import calculTemperaturesEuler

#______________________________________________________________________________________________________#
# question 4.1



def T_optimale(T_room, T_surface):
    """Calcule la température de confort selon la norme"""
    return (T_room + T_surface) / 2

def EstTemperatureOK(temps, T_room, T_surface):
    """Vérifie si la température est dans la plage de confort pendant les heures de bureau"""
    HeuresBureau = [8, 19]
    EN15251_temp = [19.5, 24]  # En °C, pas besoin de kelvin()
    
    if temps < HeuresBureau[0] or temps > HeuresBureau[1]:
        return False  # Hors des heures de bureau
    
    Temp_optimale = T_optimale(T_room, T_surface)
    return EN15251_temp[0] <= Temp_optimale <= EN15251_temp[1]

def Tmax_pour_deltaT(deltaT, T_dmax):
    """
    Fonction à annuler : T_max(deltaT) - T_dmax
    
    Arguments :
    - deltaT : durée de chauffage après 4h. (float64)
    - T_dmax : valeur cible de Tmax (ex : 24°C). (float64)
    
    Retourne :
    - Différence entre Tmax obtenu et Tmax souhaité.
    """

    t, T = calculTemperaturesEuler([0, 24], [15, 15, 15, 15, 15], 0.1, deltaT)
    T_confort = (T[0, :] + T[4, :]) / 2  # Troom = T[0], Tc2 = T[4]
    Tmax = np.max(T_confort)
    return Tmax - T_dmax



#______________________________________________________________________________________________________#
#question 4.2

def bissection_secante(f, x0, x1, tol = 0.5e-7, max_iter=50): #par défaut une tolérance de 8 décimales correctes, et un nombre d'itérations max de 50 
    '''
    Recherche de racine par dichotomie -> on coupe l'intervalle en deux et on regarde le signe de ce terme pour diminuer la taille de l'intervalle d'un facteur 1/2
    une fois que les limites de l'intervalle sont suffisamment proches de zéro, on retourne la valeur de x à cet endroit ansi qu'un code d'erreur s'il y a eu un problème.
    
    f : fonction d'entrée

    x0: début (ou fin) de l'intervalle de recherche de racine

    x1: fin (ou début) de l'intervalle de recherche de racine

    tol: tolérance de l'imprécision de la racine (par défaut 0.5e-7)

    max_iter: nombre maximal d'itérations de la recherche

    
    Sortie sous la forme:

    [x, status]

    '''
    retour = hasRoots(f,x0,x1,tol)
    if retour[1] != 0:
        return retour
    #CODE EN NEGLIGEANT LES FONCTION AVEC nbr_racines > 1  en nombre pairs ( souci pour 3 racines, 5 racines,... )
    
    
    nombre_d_iterations = math.ceil(np.log2((x1 - x0) / (2 * tol))) #arrondi supérieur

    if(nombre_d_iterations <= 0 or nombre_d_iterations > max_iter):
        return [1984,-1] # on a un souci de convergeance: un nombre négatif d'itérations...

    for i in range(nombre_d_iterations):
        x2 = (x0+x1) /2 #creation d'un point au milieu de x0 et x1 
        fx2 = f(x2)     #STOCKAGE DE L'ESTIMATION DE LA FONCTION A UN POINT X2
        fx0 = f(x0)     #STOCKAGE DE L'ESTIMATION DE LA FONCTION A UN POINT X0
        if fx2 < tol:
            return [x2 , 0] #on a trouvé la racine exacte -> on quitte
        elif(fx0 * fx2 < 0): #on compare le signe de x0 et x2 
            x1 = x2 # f(x0) et f(x2) sont de signe contraire -> le zéro se trouve entre x0 et x2
        else: 
            x0 = x2 # les deux nombres sont de même signe -> le zéro se trouve entre x2 et x1
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



#______________________________________________________________________________________________________#
# question 4.3
