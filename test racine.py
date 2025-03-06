import numpy as np
import math

def hasRoots(f, x0, x1, tol):
    ''' 
    Vérifie si la fonction possède les conditions nécessaires pour la recherche de racine :
    - Vérifie si les bornes sont bien de signe opposé
    - Intervertit les bornes si nécessaire
    - Retourne un code d'erreur si les conditions ne sont pas respectées
    '''
    fx0, fx1 = f(x0), f(x1)
    
    if tol == 0:
        return [None, -2]  # Erreur -2 : tolérance invalide
    if fx0 * fx1 > 0:
        return [None, 1]  # Erreur 1 : Pas de changement de signe, donc pas de racine unique
    if fx0 == 0:
        return [x0, 0]  # x0 est déjà une racine
    if fx1 == 0:
        return [x1, 0]  # x1 est déjà une racine
    
    return [None, 0]  # Tout va bien, on peut continuer

def bissection(f, x0, x1, tol=0.5e-7, max_iter=50):
    '''Recherche de racine par dichotomie (bisection).'''
    
    retour = hasRoots(f, x0, x1, tol)
    if retour[1] != 0:
        return retour  # Renvoie l'erreur ou la racine trouvée immédiatement
    
    fx0 = f(x0)
    
    for _ in range(max_iter):
        x2 = (x0 + x1) / 2  # Point milieu
        fx2 = f(x2)

        if abs(fx2) < tol:  # Critère d'arrêt basé sur la fonction
            return [x2, 0]
        
        if fx0 * fx2 < 0:
            x1 = x2
        else:
            x0, fx0 = x2, fx2  # Mise à jour

    return [x2, -1]  # Erreur -1 : pas de convergence après max_iter

def secante(f, x0, x1, tol=0.5e-7, max_iter=50):
    '''Recherche de racine par la méthode de la sécante.'''

    retour = hasRoots(f, x0, x1, tol)
    if retour[1] != 0:
        return retour  

    for _ in range(max_iter):
        fx0, fx1 = f(x0), f(x1)

        if abs(fx1 - fx0) < 1e-12:  # Évite la division par zéro
            return [None, -1]  # Erreur -1 : division par zéro
        
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        if abs(x2 - x1) < tol:
            return [x2, 0]  # Racine trouvée avec tolérance

        x0, x1 = x1, x2

    return [x2, -1]  # Erreur -1 : pas de convergence


def f(x):
    return x**3 - 4*x + 1  

# Intervalle de recherche
x0, x1 = -3,3
print("test bissection:")
res_bissection = bissection(f, x0, x1, tol=0.5e-7, max_iter=50) 
print(f"résultat bissection:{res_bissection} ")
print ("test sécante: ")
res_secante = secante (f,x0,x1,tol = 0.5e-7,max_iter=50)
print(f"résultat secante:{res_secante} ")

if res_bissection[1] == 0:
    print(f"Vérification: f({res_bissection[0]}) = {f(res_bissection[0])}")
if res_secante[1] == 0:
    print(f"Vérification: f({res_secante[0]}) = {f(res_secante[0])}")
