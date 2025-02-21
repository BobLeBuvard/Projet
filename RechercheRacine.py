import numpy as np
import math
def f(x):
    '''
    fonction de test pour tester si les deux fonctions de racine fonctionnent correctement. 
    Ne pas utiliser dans le code final.
    Input  la valeur de x et les coefficents dans soit une liste, soit une array
    '''
    return 5 * x**3 + 5* x**2 + 5 * x + 2 # la fonction est 5x³+5x²+ 5x +2
def hasRoots(f, x0,x1,tol):

    '''vérifie si la fonction possède les conditions nécessaires pour pouvoir trouver les racines: 
    -TODO: vérifie si la fonction possède plus de 1 racine
    -intervertit les bornes si elles sont inversées
    -vérifie si les bornes sont bien de signe opposé'''
    if (f(x0) * f(x1) > 0) or (tol == 0): # le produit des images est négatif si ils possèdent une racine entre x0 et x1 car de signe opposé (règle de l'étau) Pas bon non plus si on veut une tolérance nulle au risque de diviser par zéro
        return [1984, 1]
    # cas de base où x0 ou x1 sont racine
    elif(f(x0) == 0 ):
        return[x0,0]
    elif(f(x1) == 0) :
        return[x1,0]
    if x1<x0: #optionnel si on se goure et que x0>x1 (bornes inversées)
        x1,x0 = x0,x1
        #Le prof demande un message d'erreur ou pas ? décommenter la ligne suivante si c'est le cas, et supprimer la ligne au dessus
        # return [1984,1]
    x_vals = np.linspace(x0, x1, 50)  # echantillonne l'intervalle en 50 petites sections -> prend du temps je suis d'accord
    sign_changes = 0
    signe_precedent = np.sign(f(x_vals[0]))    #liste de signes sur l'échantillon

    for j in range(1, 50):  # Commence à 1 pour éviter l'erreur d'indexation
        signe_actuel = np.sign(f(+x_vals[j]))  # Calcule le signe actuel 
        if signe_actuel != signe_precedent:  # Vérifie le changement de signe
            sign_changes += 1
        signe_precedent = signe_actuel
        
    # Compter le nombre de changements de signe

    if sign_changes > 1 :  # Vérifier si le nombre de changements de signe est impair
        print("ok, une seule racine")
        return[0,0] #rien à dire, c'est OK
    else:
        return [1984, 1]  # Écarte les cas où il y a un nombre pair de racines
    
        #Je n'élimine pas tous les cas, mais en tout cas une bonne partie des cas avec 2+ racines en échantillonant l'espace
    
def secante(f, x0, x1, tol = 0.5e-7, max_iter=50):
    '''
    f : fonction d'entrée
    x0: début (ou fin) de l'intervalle de recherche de racine
    x1: fin (ou début) de l'intervalle de recherche de racine
    tol: tolérance de l'imprécision de la racine (par défaut 0.5e-7)
    max_iter: nombre maximal d'itérations de la recherche

    Recherche de racine par sécante -> on va créer une droite qui part de x0 et qui arrive à x2 et on va prendre son intersection avec zéro en x
    ensuite on prend cette valeur et on compare la valeur de la fonction à ce x précis avec zéro (en prenant en compte les tolérances)
    
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

def bissection(f, x0, x1, tol = 0.5e-7, max_iter=50): #par défaut une tolérance de 8 décimales correctes, et un nombre d'itérations max de 50 
    '''
     f : fonction d'entrée

    x0: début (ou fin) de l'intervalle de recherche de racine

    x1: fin (ou début) de l'intervalle de recherche de racine

    tol: tolérance de l'imprécision de la racine (par défaut 0.5e-7)

    max_iter: nombre maximal d'itérations de la recherche

    Recherche de racine par dichotomie -> on coupe l'intervalle en deux et on regarde le signe de ce terme pour diminuer la taille de l'intervalle d'un facteur 1/2
    une fois que les limites de l'intervalle sont suffisamment proches de zéro, on retourne la valeur de x à cet endroit ansi qu'un code d'erreur s'il y a eu un problème.
    
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
    
    else:    
        return [x2,0]
    
    #TODO : 
    #       Souci pour tester la convergence -> mauvais message d'erreur si divergence je crois
    #       Mentionner les racines double dans le rapport

