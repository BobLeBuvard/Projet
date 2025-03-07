import numpy as np
import math

def hasRoots(f, x0, x1, tol,sec = False):
    ''' 
    Vérifie si la fonction possède les conditions nécessaires pour la recherche de racine :
    - Vérifie si les bornes sont bien de signe opposé
    - Intervertit les bornes si nécessaire
    - Retourne un code d'erreur si les conditions ne sont pas respectées
    '''
    fx0, fx1 = f(x0), f(x1)
    
    if tol <= 0:
        print("Une tolérance égale à 0 ou négative est impossible à atteindre.")
        return[None, 1]
    if fx0 * fx1 > 0 and not sec:
        print("La fonction a le même signe aux extrémité de l'intervalle considéré, à éviter.")
        return [None, 1]  # Erreur 1 : Pas de changement de signe, donc pas de racine unique
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
        
        if fx0 * fx2 < 0:
            x1 = x2
        else:
            x0, fx0 = x2, fx2  # Mise à jour

    return [x2, -1]  # Erreur -1 : pas de convergence après max_iter

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

def hybride(f,x0,x1,tol =0.5e-7,tol_bisect = 3 ,max_iter = 30,alentours_secante = 1):
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
        print(nombre_d_iterations)
        if (nombre_d_iterations <= 0 or nombre_d_iterations > max_iter):
            print('souci de convergence, ce sera juste la méthode de la sécante')
            
            #secante brute
            for i in range(max_iter):
                if abs(fx1 - fx0) < 1e-12:  # Évite la division par zéro 1e-12 = "zéro machine"
                    return [None, -1]  # Erreur -1 : division par zéro
                
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                if abs(fx1)<= tol:
                    return [x2,0]

                x0, x1 = x1, x2
                fx0, fx1 = fx1, f(x1)
                
        else:
            fx0 = f(x0)
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
                if abs(fx1 - fx0) < 1e-12:  # Évite la division par zéro 1e-12 = "zéro machine"
                    break
                
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                if abs(fx1)<= tol:
                    return [x2,0]

                x0, x1 = x1, x2
                fx0, fx1 = fx1, f(x1)
            intervalle = abs(x0 - x1) 
            x0 = x0 + intervalle * 0.1 #on a eu un problème d'intervalle, on peut recommencer avec un x0 
            fx0 = f(x0)
        return ["fonction à la con, aucun résultat trouvé",-1]

    #partie sécante
    x0,x1 = x0_new-alentours_secante,x0_new+alentours_secante
    

    for i in range(max_iter):
        # Vérifie si la division par zéro pourrait se produire
        if abs(fx1 - fx0) < 1e-12:  # Évite la division par zéro
            return [None, -1]  # Si on évite la division par zéro, on renvoie une erreur

        # Calcul de x2 (sécante)
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        # Vérification si la solution est suffisamment proche de zéro
        if abs(fx1) <= tol or abs(x1 - x0) <= tol:  # Critère basé sur x1 ou la différence entre x1 et x0
            return [x2, 0]

        # Mise à jour de x0, x1 et des valeurs des fonctions
        x0, x1 = x1, x2
        fx0, fx1 = fx1, f(x1)

# Si on atteint le max d'itérations sans solution, on renvoie une erreur
    return [None, -1]  # Si la solution n'est pas trouvée dans le nombre d'itérations
