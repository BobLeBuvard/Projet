#CECI EST LE FICHIER GENERAL DU PROJET. ON FONCTIONNE PAR GIT COMMITS, NE FAITES PAS TROP DE COMMITS QUAND C'EST INUTILE(PAS UN PAR CHANGEMENT QUOI)
#OUBLIEZ PAS SI VOUS FAITES DES FONCTIONS DE LES COMMENTER AVEC DES COMMENTAIRES COMPLETS: EX:

def fonctionRandom():
    '''CECI EST LA ZONE DE COMMENTAIRES: DITES COMMENT UTILISER LA FONCTION ET CE QU'ELLE FAIT,ET CE QU'ELLE RETOURNE:
    exemple: cette fonction imprime 10 sans rien retourner, et sans rien en entrée
    '''
print(10)


print("Bissectrice")

import numpy as np
    
def f(x, coeffs):
    return coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]

def bisection_method(coeffs, a, b, tol=1e-6):
    if f(a, coeffs) * f(b, coeffs) >= 0:
        raise ValueError("The function must have opposite signs at a and b.")
    
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c, coeffs) == 0:
            return c
        elif f(c, coeffs) * f(a, coeffs) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2

# Example usage
coeffs = list(map(float, input("Enter coefficients (a b c d) for ax^3 + bx^2 + cx + d: ").split()))
a, b = map(float, input("Enter the interval [a, b]: ").split())

root = bisection_method(coeffs, a, b)
print(f"The root found is: {root}")
