import matplotlib.pyplot as plt
from RechercheRacine import bissection
import numpy as np
from SimTABS import calculTemperaturesEuler,kelvin,celsius,cycles_apres_convergence,dessinemoassa, debug

def fonctiondroite(hauteur, label = None):
    '''fonction qui va plot y = 0 sur le graphique'''
    if debug: plt.plot(np.arange(25),np.zeros(25) + hauteur , label = label)

#______________________________________________________________________________________________________#
# question 4.1


def T_max(delta_t,  T0 = kelvin(np.array([15,15,15,15,15])) , no_max = False):
    '''
    Fonction qui calcule le maximum de température de confort d'un cycle (avec un delta T donné)
    
    Fonction à annuler : T_max(deltaT) - T_dmax 

    IN:

    - deltaT : durée de chauffage après 4h. (float64)

    - T_dmax : valeur cible de Tmax (ex : 24°C). (float64)
    
    OUT:
    
    - Différence entre Tmax obtenu et Tmax souhaité.
    '''
    MAX = 0
    t, T = calculTemperaturesEuler([0, 24], T0,  0.01,num_du_scenario = 4, delta_t = delta_t)
    T_confort = (T[0, :] + T[4, :]) / 2  # T_room = T[0], T_c2 = T[4]
    if no_max == False: 
        MAX = np.max(T_confort)
        
        return MAX, t, T_confort#si on ne veut pas de max --> no_max=True ben on ne le calcule pas.
    print(MAX)
    print(no_max)
    return None,t,T_confort

def question_4_1(delta_t,T_max_d):
    '''
    
    fonction qui dessine le max de température de confort en apellant T_max et en plottant les températures adéquates.
    
    T_max_d en degrés celsius
    
    delta_t en heures
    
    '''
    MAX,t,T_confort = T_max(delta_t)
    if debug: print(celsius(MAX)) #print la température max sur l'intervalle en degrés celsius
    plt.xlabel("temps (24h)", fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.ylabel("température optimale ", fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.title(label = f'Température de confort sur 24h -> delta_t = {delta_t}')
    plt.plot(t,celsius(T_confort) ,label= "température de confort")
    fonctiondroite(T_max_d, label = 'température préférée')
    plt.plot(4,T_max_d,'.', label = f'début de la période de chauffe ({4}h)')
    plt.plot(4+delta_t,T_max_d,'.', label = f'fin de la période de chauffe ({4+delta_t}h)')
    
    plt.legend( loc='best')
    plt.show()
    return 0
 

#______________________________________________________________________________________________________#
#question 4.2

def recherche_delta_t (T_max_d, intervalle = [0,24], tol = 0.5e-4, T0 = kelvin(np.array([15, 15, 15, 15, 15])),no_max = False):
    '''
        fonction qui va rechercher le delta_t tel que l'on ne dépassera jamais T_max_d sur un cycle de 24h

        IN:
    
        T_max_d (float)--> Température maximale désirée en Kelvin.
        
        intervalle (list[float, float])--> Intervalle de recherche pour delta_t (par défaut [0, 24]).
        
        tol (float)--> Tolérance pour la convergence de la méthode de la bissection (par défaut 0.5e-7).
        
        T0 (numpy.ndarray)--> Tableau numpy des températures initiales en Kelvin (par défaut [288.15, 288.15, 288.15, 288.15, 288.15]).

        OUT:

        delta_t (float)--> Période delta_t nécessaire pour ne pas dépasser T_max_d.
        '''


    f_difference = lambda deltaT: T_max(deltaT, T0,no_max = no_max)[0] - T_max_d 
    '''
    fonction qui fait la différence entre T_max qui varie en fonction de delta et T_max_d qui est choisis abritrairement, il faut en 
    rechercher la racine pour pouvoir trouver delta_t
    '''
    
    delta_t ,statut = bissection(f_difference,intervalle[0],intervalle[1], tol=tol, max_iter=54)
    
    if statut !=0 : 
        print('erreur, problème de racine') 
        return(statut)
    return delta_t

def question_4_2(T_max_d):
    '''T_max_d en degrés celsius'''
    delta_t = recherche_delta_t(kelvin(T_max_d))
    print(f'on a trouvé un delta_t approchant la T_d désirée: {delta_t} heures')
    question_4_1(delta_t,T_max_d)
    
#______________________________________________________________________________________________________#
# question 4.3



#EN15251 est une array contenant t0 et tf et Tmin et Tmax -> [8,19,19.5,24]
EN15251 = np.array([8,19,19.5,24])
def verification_EN15251(delta_t,EN15251,T0 = kelvin(np.array([15,15,15,15,15])) ):
    MAX,t,T_confort  = T_max(delta_t,T0 = T0 ,no_max = True)
    plt.plot(t,celsius(T_confort), label = 'température de confort')
    for i in range(2): 
        fonctiondroite(EN15251[i+2], label = ['température minimale','température maximale'][i])
    plt.title(label = "graphique de la température de confort pendant la dernière journée")
    plt.xlabel('heures de la journée (h)')
    plt.ylabel('température (°C')
    plt.show()
    for i in range(len(t)):
        T_confort_i = T_confort[i]
        if (EN15251[0]<= t[i]<=  EN15251[1]):
            if not(EN15251[2] < T_confort_i < EN15251[3]): #est ce que les extrémas sont pris avec ?
                print("La norme EN15251 n'est pas respectée.")
                return False
    print("La norme EN15251 est respectée.")
    return True

def question_4_3(T_max_d, EN15251 = np.array([8,19,19.5,24]), T0 = kelvin(np.array([15, 15, 15, 15, 15]))):
    
    h = 0.01
    FenetreDeTemps = [0,24]
    delta_t = recherche_delta_t(kelvin(T_max_d),T0 = T0) #OK
    print(f"on a trouvé un delta_t de {delta_t}")
    days_to_converge, T0_new = cycles_apres_convergence(T0,FenetreDeTemps,h,delta_t= delta_t,num_du_scenario=4 ) #T0_new est les conditions initiales du dernier jour
    if days_to_converge == None:
        print("Les températures ne se sont pas stabilisées.")
        return -1
    plt.show()
    # t,T = calculTemperaturesEuler(FenetreDeTemps,T0_new,h,num_du_scenario= 4,delta_t= delta_t) #calculer le dernier jour UNIQUEMENT POUR PLOT
    # dessinemoassa(t,celsius(T),['T_room','T_t','T_cc','T_c1','T_c2'],xlabel='temps (h)',ylabel='température (°C)',titre= f'températures au jour {days_to_converge}')
    
    return verification_EN15251(delta_t,EN15251,T0_new)


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



OUI
    1) 
    2) 
    3) vérifier la norme

NON
1) trouver un  delta_t qui est ok avec le premier jour
2) vérifier la stabilisation
3) vérifier si le dernier jour stabilisé est OK avec les conditions EN15251
    



