import numpy as np
import math
from config import debug

def hasRoots(f, x0, x1, tol, sec=False):
    ''' 
    Vérifie si la fonction possède les conditions nécessaires pour la recherche de racine :
    - Vérifie si les bornes sont bien de signe opposé (pour la bissection)
    - Vérifie si x0 ou x1 est déjà une racine
    - Retourne un statut 1 si les conditions ne sont pas respectées
    '''
    if not isinstance(x0, (int, float)) or not isinstance(x1, (int, float)): #vérifier que x0 et x1 existent bel et bien
        return ["x0 ou x1 ne sont pas définis",1]
    fx0, fx1 = f(x0), f(x1)

    if tol <= 0 or abs(x0 - x1) <= tol:
        print("Erreur : Tolérance invalide ou intervalle trop petit.")
        return [None, 1]  # ERREUR (statut 1)

    if abs(fx0) <= tol:
        print(f"Racine trouvée immédiatement : {x0}")
        return [x0, 0]  # x0 est une racine

    if abs(fx1) <= tol:
        print(f"Racine trouvée immédiatement : {x1}")
        return [x1, 0]  # x1 est une racine

    if not sec and fx0 * fx1 > 0:  # Vérification pour la bissection
        print("Erreur : Pas de changement de signe (pas de racine unique garantie).")
        return [None, 1]  # ERREUR (statut 1) pour respecter l'énoncé

    return [None, 0]  # Tout va bien, on peut continuer


def bissection(f, x0, x1, **kwargs):
    tol = kwargs.get('tol_rac',0.5e-7)
    max_iter = kwargs.get('max_iter',50)
    '''Recherche de racine par dichotomie (bissection).'''
    
    
    retour = hasRoots(f, x0, x1, tol)
    if retour[1] != 0 or retour[0] is not None:
        return retour  # Renvoie l'erreur ou la racine trouvée immédiatement

    # Éviter une erreur si l'intervalle est trop petit
    if x1 - x0 <= 2 * tol:
        print("Intervalle initial trop petit pour la bissection.")
        return [None, -1]

    nombre_d_iterations = math.ceil(np.log2((x1 - x0) / (2 * tol)))
    if nombre_d_iterations < 1 or nombre_d_iterations > max_iter:
        print("Problème de convergence : trop d'itérations nécessaires.")
        return [None, -1]  # Pas de convergence
    
    fx0 = f(x0)
    for _ in range(nombre_d_iterations):
        x2 = (x0 + x1) / 2  # Point milieu
        fx2 = f(x2)

        if abs(fx2) <= tol:
            return [x2, 0]
        
        if fx0 * fx2 < 0:
            x1 = x2
        else:
            x0, fx0 = x2, fx2
    
    print("La méthode de la bissection n'a pas convergé.")
    return [None, -1]





def secante(f, x0, x1,**kwargs):
    '''Recherche de racine par la méthode de la sécante.'''
    tol = kwargs.get('tol_rac',0.5e-7)
    max_iter = kwargs.get('max_iter',50)
    retour = hasRoots(f, x0, x1, tol, sec=True)
    if retour[1] != 0 or retour[0] is not None:
        return retour  

    fx0, fx1 = f(x0), f(x1)

    for i in range(max_iter):
        if abs(fx1 - fx0) < 1e-12:  # Évite la division par zéro
            print("Problème de convergence : division par zéro.")
            return [None, 1]  

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)

        if abs(fx2) <= tol:
            return [x2, 0]

        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2

    print("La méthode de la sécante n'a pas convergé.")
    return [None, -1]

def secante_precalculee(f,x0,x1,tol,fx0,fx1,max_iter):
    for i in range(max_iter):
        if abs(fx1)<= tol:
            return [x2,0]
        if abs(fx1 - fx0) < 1e-12 or  abs(x1 - x0) < tol:  # Évite la division par zéro 1e-12 = "zéro machine"
            print("division par zéro")
            print(f'{fx0} {fx1}')
            return [None, -1]  # Erreur -1 : division par zéro
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        x0, x1 = x1, x2
        fx0, fx1 = fx1, f(x1)
    print('pas de convergence de la secante')
    return [None,-1]
def hybride(f,x0,x1,**kwargs):
    tol = kwargs.get('tol_rac',0.5e-7)
    max_iter = kwargs.get('max_iter',30)
    tol_bisect = kwargs.get('tol_bisect_hybride',0.1)
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

                    return [x0_new,0]
                
                if fx0 * fx2 < 0:
                    x1 = x2
                else:
                    x0, fx0 = x2, fx2  # Mise à jour
            
        
    else:
        #Partie secante.
        print("La fonction a le même signe aux extrémité de l'intervalle considéré, à éviter. Passage direct à la sécante") 
        for i in range(max_iter):
            for j in range(max_iter):
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
    