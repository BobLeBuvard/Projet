import numpy as np
import math

def hasRoots(f, x0,x1,tol):

    '''vérifie si la fonction possède les conditions nécessaires pour pouvoir trouver les racines: 
    -intervertit les bornes si elles sont inversées
    -vérifie si les bornes sont bien de signe opposé'''
    # Vérification des conditions
    fx0 = f(x0)
    fx1 = f(x1)
    if ( fx0 * fx1 > 0) or (tol == 0):
        return [1984, 1]
    elif fx0 == 0:
        return [x0, 0]
    elif fx1 == 0:
        return [x1, 0]
    
    if x1 < x0:
        x1, x0 = x0, x1
    
    # Ajoute un return par défaut pour éviter None
    return [1984, 0] 
 
def bissection(f, x0, x1, tol = 0.5e-7, max_iter=50): #par défaut une tolérance de 8 décimales correctes, et un nombre d'itérations max de 50 
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
    
    nombre_d_iterations = math.ceil(np.log2((x1 - x0) / (2 * tol))) #arrondi supérieur

    if(nombre_d_iterations <= 0 or nombre_d_iterations > max_iter):
        return [1984,-1] # on a un souci de convergence: un nombre négatif d'itérations... ou simplement trop

    for i in range(nombre_d_iterations):
        x2 = (x0+x1) /2 #creation d'un point au milieu de x0 et x1 
        fx2 = f(x2)     #STOCKAGE DE L'ESTIMATION DE LA FONCTION A UN POINT X2
        fx0 = f(x0)     #STOCKAGE DE L'ESTIMATION DE LA FONCTION A UN POINT X0
        if abs(fx2) < tol:
            return [x2 , 0] #on a trouvé la racine exacte -> on quitte
        elif(fx0 * fx2 < 0): #on compare le signe de x0 et x2 
            x1 = x2 # f(x0) et f(x2) sont de signe contraire -> le zéro se trouve entre x0 et x2
        else: 
            x0 = x2 # les deux nombres sont de même signe -> le zéro se trouve entre x2 et x1
        print(f"nouveau x2 trouvé dans la recherche de racine: {x2}")
   
    return [x2,0]
   
def secante(f, x0, x1, tol = 0.5e-7, max_iter=50):
    '''
    Recherche de racine par sécante -> on va créer une droite qui part de x0 et qui arrive à x2 et on va prendre son intersection avec zéro en x
    ensuite on prend cette valeur et on compare la valeur de la fonction à ce x précis avec zéro (en prenant en compte les tolérances)

    f : fonction d'entrée

    x0: début (ou fin) de l'intervalle de recherche de racine

    x1: fin (ou début) de l'intervalle de recherche de racine

    tol: tolérance de l'imprécision de la racine (par défaut 0.5e-7)

    max_iter: nombre maximal d'itérations de la recherche
    
    Sortie sous la forme:

    [x, status]  
    '''
    retour = hasRoots(f,x0,x1,tol) #fonction qui détermine si la fonction a plus d'une racine
    if retour[1] != 0:
        return retour   # la fonction a plus d'une racine

    for i in range(max_iter):
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