#CECI EST LE FICHIER GENERAL DU PROJET. ON FONCTIONNE PAR GIT COMMITS, NE FAITES PAS TROP DE COMMITS QUAND C'EST INUTILE(PAS UN PAR CHANGEMENT QUOI)
print("Bissectrice")

import numpy as np

def ajouter10 (machin):
    '''COMMENT SECTION DE LA FONCTION: METTRE ENTRE 3 guillemets l'explication de la fonction: EX:
        ceci est une fonction qui rajoute 10 à la variable entrée'''
    machin += 10
    return 10
    
    
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
