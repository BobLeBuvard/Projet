import numpy as np
import scipy as scp

def f(x):
    '''
    fonction de test pour tester si les deux fonctions de racine fonctionnent correctement. 
    Ne pas utiliser dans le code final.
    Input  la valeur de x et les coefficents dans soit une liste, soit une array
    '''
    return 5 * x**3 + 5* x**2 + 5 * x + 2 # la fonction est 5x³+5x²+ 5x +2

def secante(f, x0, x1, tol):

    return [x, statut]

def bissection(f, x0, x1, tol = 0.5e-7): #par défaut une tolérance de 8 décimales correctes
    '''Recherche de racine par dichotomie -> on coupe et on regarde si on se rapproche.
    On part du principe que f est continue sur cet intervalle
    Attention que x0 et x1 peuvent être tous les deux positifs dans certains cas ( ex dans le cas de (x-1)² en x = 0 et x = 2 tous les 2 valent y = 1 mais il existe une racine)
    Je suppose qu'on doit prendre en compte les fonctions qui ont plus de 1 racine, c'est pourquoi j'ai émis l'hypothèse au dessus
    
    on s'arrête après le nombre d'itérations suffisantes pour rentrer dans les tolérances demandées -> c'est combien ?'''
    
    
    #CODE EN NEGLIGEANT LES FONCTION AVEC nbr_racines > 1  en nombre pairs ( souci pour 3 racines, 5 racines,... )
    if (f(x0) * f(x1) > 0) or (tol == 0): # le produit des images est négatif si ils possèdent une racine entre x0 et x1 car de signe opposé (règle de l'étau) Pas bon non plus si on veut une tolérance nulle au risque de diviser par zéro
        return [1984, 1]
    #cas de base où x0 ou x1 sont racine
    elif(f(x0) == 0 ):
        return[x0,0]
    elif(f(x1) == 0) :
        return[x1,0]
    
    if x1<x0: #optionnel si on se goure et que x0>x1 (bornes inversées)
        temp = x1
        x1 = x0 
        x0 = temp
        #Le prof demande un message d'erreur ou pas ? décommenter la ligne suivante si c'est le cas, et supprimer les lignes au dessus
        # return [1984,1]

    nombre_d_iterations = round(np.log2((x1 - x0) / (2 * tol))) + 1 #arrondi supérieur
    print(nombre_d_iterations)
    if(nombre_d_iterations <= 0 or nombre_d_iterations>80):
        return [1984,-1] # on a un souci de convergeance: un nombre négatif d'itérations...

    for i in range(nombre_d_iterations):
        x2 = (x0+x1) /2 #creation d'un point au milieu de x0 et x1 
        if f(x2) ==0:
            return [x2 , 0] #on a trouvé la racine exacte -> on quitte
        elif(f(x0) * f(x2) < 0): #on compare le signe de x0 et x2 
            x1 = x2 # f(x0) et f(x2) sont de signe contraire -> le zéro se trouve entre x0 et x2
        else: 
            x0 = x2 # les deux nombres sont de même signe -> le zéro se trouve entre x2 et x1
    
    else:    
        return [x2,0]
    
    #TODO : petit souci pour les fonctions avec plusieurs racines (cas négligé) 
    #       Souci pour tester la convergence -> mauvais message d'erreur si divergence je crois


