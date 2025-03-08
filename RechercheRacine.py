import numpy as np
import math
from config import debug

def hasRoots(f, x0, x1, tol,sec = False):
    ''' 
    Vérifie si la fonction possède les conditions nécessaires pour la recherche de racine :
    - Vérifie si les bornes sont bien de signe opposé
    - Intervertit les bornes si nécessaire
    - Retourne un code d'erreur si les conditions ne sont pas respectées
    '''
    fx0, fx1 = f(x0), f(x1)
    
    if tol <= 0 or abs(x0- x1)<= tol :
        print("Une tolérance égale à 0 ou négative est impossible à atteindre.")
        return[fx0, 1]
    if fx0 * fx1 > 0 and not sec:
        print("La fonction a le même signe aux extrémité de l'intervalle considéré, à éviter.")
        return [None, -1]  # Erreur 1 : Pas de changement de signe, donc pas de racine unique
    if abs(fx0) <= tol:
        print(f"La solution est {x0}")
        return [x0, 0]  #x0 est déjà une racine
    
    if abs(fx1) <= tol:
        print(f"La solution est {x1}")
        return [x1, 0]  # x1 est déjà une racine
    
    return [None, 0]  # Tout va bien, on peut continuer

def bissection(f, x0, x1, tol=0.5e-7, max_iter=50):
    '''Recherche de racine par dichotomie (bissection).'''
    

    retour = hasRoots(f, x0, x1, tol)
    if retour[1] != 0 or retour[0] != None:
        return retour  # Renvoie l'erreur ou la racine trouvée immédiatement
    
    
    nombre_d_iterations = math.ceil(np.log2((x1 - x0) / (2 * tol))) #arrondi supérieur

    if(nombre_d_iterations <= 0 or nombre_d_iterations > max_iter):
        return [1984,-1] # on a un souci de convergence: un nombre négatif d'itérations... ou simplement trop
    fx0 = f(x0)
    for _ in range(nombre_d_iterations):
        x2 = (x0 + x1) / 2  # Point milieu
        fx2 = f(x2)

        if abs(fx2) < tol:  # Critère d'arrêt basé sur la fonction
            return [x2, 0]
        
        if fx0 * fx2 <= 0:
            x1 = x2
        else:
            x0, fx0 = x2, fx2  # Mise à jour
            # if debug : print(f"Nouveau x2 trouvé dans la recherche de racine: {x2}")
    return [x2, 0]




def secante(f, x0, x1, tol=0.5e-7, max_iter=65):
    '''Recherche de racine par la méthode de la sécante.'''

    retour = hasRoots(f, x0, x1, tol,sec = True)
    if retour[1] != 0 or retour[0] != None:
        return retour  
    fx0, fx1 = f(x0), f(x1)
    for i in range(max_iter):
        if abs(fx1 - fx0) < 1e-12:  # Évite la division par zéro 1e-12 = "zéro machine"
            return [None, -1]  # Erreur -1 : division par zéro
        
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(fx1)<= tol:
            return [x2,0]

        x0, x1 = x1, x2
        fx0, fx1 = fx1, f(x1)

    return [x2, -1]  # Erreur -1 : pas de convergence
def secante_precalculee(f,x0,x1,tol,fx0,fx1,max_iter):
    for i in range(max_iter):
        if abs(fx1)<= tol:
            return [x2,0]
        if abs(fx1 - fx0) < 1e-12 or  abs(x1 - x0) < tol:  # Évite la division par zéro 1e-12 = "zéro machine"
            print("division par zéro")
            print(f'{fx0} {fx1}')
            return [None, -1]  # Erreur -1 : division par zéro
        x2 = x1 - fx1 * (x1 - x0) / fx1 - fx0

        x0, x1 = x1, x2
        fx0, fx1 = fx1, f(x1)
    print('pas de convergence de la secante')
    return [None,-1]
def hybride(f,x0,x1,tol =0.5e-7,tol_bisect = 0.1 ,max_iter = 30):
    #vérif conditions initiales
    fx0, fx1 = f(x0), f(x1)
    
    if tol <= 0:
        print("Une tolérance égale à 0 ou négative est impossible à atteindre.")
        return[None, 1]
    
    if abs(fx0) <= tol:
        print(f"La solution est {x0}")
        return [x0, 0]  # x0 est déjà une racine
    
    if abs(fx1) <= tol:
        print(f"La solution est {x1}")
        return [x1, 0]  # x1 est déjà une racine
    

    #partie bissection:
    if not fx0 * fx1 > 0:
        nombre_d_iterations = math.ceil(np.log2((x1 - x0) / (2 * tol_bisect))) #arrondi supérieur
        if debug: print(nombre_d_iterations)
        if (nombre_d_iterations <= 0 or nombre_d_iterations > max_iter):
            print('souci de convergence, ce sera juste la méthode de la sécante')
            racine,statut = secante_precalculee(f, x0, x1, tol, fx0, fx1, max_iter)
            #secante brute
            return[racine,statut]
        else:
            #fx0 = f(x0)
            for _ in range(max_iter):
                x2 = (x0 + x1) / 2  # Point milieu
                fx2 = f(x2)

                if abs(fx2) < tol_bisect:  # Critère d'arrêt basé sur la fonction
                    x0_new = x2
                    break
                
                if fx0 * fx2 < 0:
                    x1 = x2
                else:
                    x0, fx0 = x2, fx2  # Mise à jour
            
        
    else:
        print("La fonction a le même signe aux extrémité de l'intervalle considéré, à éviter. Passage direct à la sécante") 
        for k in range(max_iter):
            for i in range(max_iter):
                racine,statut = secante_precalculee(f, x0, x1, tol, fx0, fx1, max_iter)
            if not statut:
                return[racine,statut]
            intervalle = abs(x0 - x1) 
            x0 = x0 + intervalle * 0.1 #on a eu un problème d'intervalle, on peut recommencer avec un x0 
            fx0 = f(x0)
        return ["fonction à la con, aucun résultat trouvé",-1]

    #partie sécante
    
    racine,statut = secante(f, x0, x1,tol= tol, max_iter=30)
    return[racine,statut]
    