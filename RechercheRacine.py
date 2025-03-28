import numpy as np
def hasRoots(f, x0, x1, tol, sec=False):
    
    if not isinstance(x0, (int, float)) or not isinstance(x1, (int, float)): #vérifier que x0 et x1 existent bel et bien
        return ["x0 ou x1 ne sont pas définis",1]
    fx0, fx1 = f(x0), f(x1)

    if tol <= 0 or abs(x0 - x1) <= tol:
        print("Erreur : Tolérance invalide ou intervalle trop petit.")
        return [None, 1]  

    if abs(fx0) <= tol:
        print(f"Racine trouvée immédiatement : {x0}")
        return [x0, 0]  # x0 est une racine

    if abs(fx1) <= tol:
        print(f"Racine trouvée immédiatement : {x1}")
        return [x1, 0]  # x1 est une racine

    if not sec and fx0 * fx1 > 0:  # Vérification pour la bissection
        print("Erreur : Pas de changement de signe (pas de racine unique garantie).")
        return [None, 1]  

    return [None, 0]  


def bissection(f, x0, x1, **kwargs):

    # Initialisation des variables
    tol = kwargs.get('tol_rac',0.5e-7)
    max_iter = kwargs.get('max_iter',50)
        
    # Calcul
    retour = hasRoots(f, x0, x1, tol)
    if retour[1] != 0 or retour[0] is not None:
        return retour  # Renvoie l'erreur ou la racine trouvée immédiatement

    # Éviter une erreur si l'intervalle est trop petit
    if x1 - x0 <= 2 * tol:
        print("Intervalle initial trop petit pour la bissection.")
        return [None, -1]

    nombre_d_iterations = round(np.log2((x1 - x0) / (2 * tol))) +1
    if nombre_d_iterations < 1 or nombre_d_iterations > max_iter:
        print("Problème de convergence : trop d'itérations nécessaires.")
        return [None, -1]  # Pas de convergence
    
    fx0 = f(x0)
    for _ in range(nombre_d_iterations):
        x2 = (x0 + x1) / 2  
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

    # Initialisation des variables
    tol = kwargs.get('tol_rac',0.5e-7)
    max_iter = kwargs.get('max_iter',50)

    # Calcul
    retour = hasRoots(f, x0, x1, tol, sec=True)
    if retour[1] != 0 or retour[0] is not None:
        return retour  

    fx0, fx1 = f(x0), f(x1)

    for i in range(max_iter):
        if abs(fx1 - fx0) < 1e-12:  # Évite la division par zéro
            print("Problème de convergence : division par zéro.")
            return [None, 1]  

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0) # sécante
        fx2 = f(x2)

        if abs(fx2) <= tol:
            return [x2, 0]

        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2

    print("La méthode de la sécante n'a pas convergé.")
    return [None, -1]