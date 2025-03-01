import matplotlib.pyplot as plt
from RechercheRacine import bissection
import numpy as np
from SimTabsFinal import calculTemperaturesEuler,kelvin,celsius,cycles_apres_convergence,dessinemoassa, debug

def fonctiondroite(hauteur):
    '''fonction qui va plot y = 0 sur le graphique'''
    if debug: plt.plot(np.arange(25),np.zeros(25) + hauteur )

#______________________________________________________________________________________________________#
# question 4.1


def T_max(delta_t, no_max = False, T0 = None):
    """
    Fonction qui calcule le maximum de température de confort d'un cycle (avec un delta T donné)
    
    Fonction à annuler : T_max(deltaT) - T_dmax 

    IN:

    - deltaT : durée de chauffage après 4h. (float64)

    - T_dmax : valeur cible de Tmax (ex : 24°C). (float64)
    
    OUT:
    
    - Différence entre Tmax obtenu et Tmax souhaité.
    """
    MAX = 0
    t, T = calculTemperaturesEuler([0, 24], T0,  0.01,num_du_scenario = 4, delta_t = delta_t)
    T_confort = (T[0, :] + T[4, :]) / 2  # T_room = T[0], T_c2 = T[4]
    if no_max == False:
        MAX = np.max(T_confort) #si on ne veut pas de max --> no_max=True ben on ne le calcule pas.
    return MAX,t,T_confort

def question_4_1(delta_t,T_max_d):

    #TODO: Docstring de la fonction (description, commentaires)

    MAX,t,T_confort = T_max(delta_t)
    if debug: print(celsius(MAX)) #print la température max sur l'intervalle en degrés celsius
    plt.xlabel("temps (24h)", fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.ylabel("température optimale", fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.title(label = f'Température de confort sur 24h -> delta_t = {delta_t}')
    plt.plot(t,celsius(T_confort)-T_max_d ,label= "prout")
    fonctiondroite(0)
    plt.legend( loc='best')
    plt.show()
 

#______________________________________________________________________________________________________#
#question 4.2

def recherche_delta_t (T_max_d, intervalle = [0,24], tol = 0.5e-7, T0 = kelvin(np.array([15, 15, 15, 15, 15]))):
    '''fonction qui va rechercher le delta_t tel que l'on ne dépassera jamais T_max_d sur un cycle de 24h'''
    #TODO: ajouter les in et out de la fontion (avec leurs type associé (int, float, array, list, etc...)) sous la forme :
    '''
    EXEMPLE

    IN:
    
    T_max -> JESAISPAS

    OUT:

    delta_t -> JE SAIS PAS NON PLUS

    Espacer entre les lignes pour une mise à la ligne. 
    '''


    f_difference = lambda deltaT: T_max(deltaT, T0)[0] - T_max_d 
    '''
    fonction qui fait la différence entre T_max qui varie en fonction de delta et T_max_d qui est choisis abritrairement, il faut en 
    rechercher la racine pour pouvoir trouver delta_t
    '''
    delta_t ,statut = bissection(f_difference,intervalle[0],intervalle[1], tol=tol, max_iter=54)
    #TODO : expliciter les messages d'erreur -> des messages clairs
    if statut !=0 : 
        print('erreur, problème de racine') 
        return(-1)
    return delta_t


#______________________________________________________________________________________________________#
# question 4.3



#EN15251 est une array contenant t0 et tf et Tmin et Tmax -> [8,19,19.5,24]
EN15251 = np.array([8,19,19.5,24])
def verification_EN15251(delta_t,EN15251):
    MAX,t,T_max_arr = T_max(delta_t,no_max = True)
    T_confort = T_max_arr
    
    for i in range(len(t)):
        T_confort_i = T_confort[i]
        if (EN15251[0]<= t[i]<=  EN15251[1]):
            if not(EN15251[2] < T_confort_i < EN15251[3]): 
                print("La norme EN15251 n'est pas respectée.")
                return False
    print("La norme EN15251 est respectée.")
    return True

def question_4_3(T_max_d,EN15251, T0 = kelvin(np.array([15, 15, 15, 15, 15]))):
    h = 0.01
    
    FenetreDeTemps = [0,24]
    days_to_converge, T0_new = cycles_apres_convergence(T0,FenetreDeTemps,h) #T0_new est les conditions initiales du dernier jour
    if days_to_converge == None:
        print("Les températures ne se sont pas stabilisées.")
        return -1
    delta_t = recherche_delta_t(T_max_d,T0 = T0_new)
    t,T = calculTemperaturesEuler(FenetreDeTemps,T0_new,h,num_du_scenario= 4,delta_t= delta_t) #calculer le dernier jour UNIQUEMENT POUR PLOT
    for i in range(2): fonctiondroite(EN15251[i])
    dessinemoassa(t,celsius(T),['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='temps (h)',ylabel='température (°C)',titre= 'températures au jour {days_to_converge}')
    return verification_EN15251(delta_t,EN15251)


'''commentaire supplémentaire: 

on demande à un moment:

"Le système de pilotage doit être conçu de
telle sorte que l'on puisse (1) chauffer/refroidir/mettre à l'arrêt en fonction d'un programme
de 24h donné ou (2) démarrer le chauffage/refroidissement/mise `a l'arrêt en fonction de
certaines variables, par exemple une température de confort."

pour pouvoir faire cela, on a plusieurs méthodes: Soit on rajoute un paramètre dans calculTemperaturesEuler qui s'appelle Force_heating,
soit on peut utiliser le numéro de sénario 5 et jouer avec le paramètre delta_t le scénario adéquat.
On ne peut pas changer correctement à la volée car CalculTempératuresEuler calcule sur une fenêtre de températures. 
Ce qu'on pourrait faire par contre c'est à l'aide du paramètre delta_t et Force_heating anticiper ce qu'on voudrait 
en fonction des conditions du jour précédent, et faire un scénario adapté. Ensuite il faudrait d'ailleurs 

#TODO modifier la fonction cycles_apres_convergence() pour qu'elle et CalculTemperaturesEuler puissent se servir de ce paramètre

'''