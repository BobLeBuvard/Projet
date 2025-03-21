import matplotlib.pyplot as plt
from RechercheRacine import bissection, hybride,secante
import numpy as np
from SimTABS import*
import time

def fonctiondroite(hauteur, label = None):
    global FenetreDeTemps
    '''fonction qui va plot y = 0 sur le graphique'''
    plt.plot(np.arange(FenetreDeTemps[1]-FenetreDeTemps[0]+1),np.zeros(FenetreDeTemps[1]-FenetreDeTemps[0]+1) + hauteur , label = label)


#______________________________________________________________________________________________________#
# question 4.1


def T_max(delta_t, **kwargs):
    '''
    Fonction qui calcule le maximum de température de confort d'un cycle (avec un delta T donné)
    
    Fonction à annuler : T_max(delta_t) - T_dmax 

    IN:

    delta_t -> durée de chauffage après 4h. (float64)

    **kwargs -> se référer aux arguments qu'on peut mettre
    
    OUT:
    
    - Différence entre Tmax obtenu et Tmax souhaité.
    '''
    global h,T0
    kwargs['delta_t'] = delta_t #ajout aux arguments si ce n'est pas déjà le cas
    T0 =kwargs.pop('T0',T0)
    no_max = kwargs.pop('no_max', False)
    h = kwargs.pop('h',h)
    global FenetreDeTemps
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',FenetreDeTemps)
    MAX = 0

    t, T = calculTemperaturesEuler(FenetreDeTemps, T0,h,**kwargs)
    T_confort = (T[0, :] + T[4, :]) / 2  # T_room = T[0], T_c2 = T[4]
    if no_max: 
        return t,T_confort
    MAX = np.max(T_confort)
    return MAX, t, T_confort#si on ne veut pas de max --> no_max=True ben on ne le calcule pas.


def question_4_1(**kwargs):
    '''
    
    fonction qui dessine le max de température de confort en apellant T_max et en plottant les températures adéquates.
    
    delta_t en heures
    
    '''
    delta_t = kwargs.pop('delta_t',0)
    T_max_d = kwargs.pop('T_max_d',0)
    MAX,t,T_confort = T_max(delta_t, **kwargs)
    indice_max = np.argmax(T_confort)

# Récupérer le temps t correspondant à ce maximum
    t_max = t[indice_max]
    t_max_seconde = t_max * 3600
    print(f"maximum de température de confort: {MAX}°C à t = {t_max_seconde}s, c'est à dire {t_max}h")#print la température max sur l'intervalle en degrés celsius
    plt.xlabel(f"temps ({FenetreDeTemps[1] - FenetreDeTemps[0]}h)") # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.ylabel("température optimale ") # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.title(label = f'Température de confort sur {FenetreDeTemps[1]-FenetreDeTemps[0]}h -> delta_t = {delta_t}')
    plt.plot(t,T_confort ,label= "température de confort")
    fonctiondroite(T_max_d, label = 'T_max_d')
    plt.plot(4,T_max_d,'.', label = f'début de la période de chauffe ({4}h)')
    plt.plot(4+delta_t,T_max_d,'.', label = f'fin de la période de chauffe ({4+delta_t}h)')
    
    plt.legend( loc='best')
    plt.show()
    return MAX
 

#______________________________________________________________________________________________________#
#question 4.2
def recherche_delta_t (T_max_d,**kwargs):
    '''
        fonction qui va rechercher le delta_t tel que l'on ne dépassera jamais T_max_d sur un cycle de 24h

        IN:
    
        T_max_d (float) -> Température maximale désirée en celsius.
        
        **kwargs -> se référer aux arguments qu'on peut mettre
        
         OUT:

        delta_t (float) -> Période delta_t nécessaire pour ne pas dépasser T_max_d.
        '''

    
    kwargs.pop('delta_t',None) #on efface le delta_t des kwargs pour la recherche 
    kwargs['T_max_d'] = T_max_d #on ajoute T_max_d aux arguments pour T_max
    kwargs['num_du_scenario'] = 4
    kwargs['tol_rac'] = kwargs['h']# on ne peut pas avoir une précision parfaite à cause du h choisi impossible d'avoir plus précis que h (déterminé par essai erreur).
    f_difference = lambda delta_t: T_max(delta_t,**kwargs)[0] - T_max_d 
    '''
    fonction qui fait la différence entre T_max qui varie en fonction de delta et T_max_d qui est choisie abritrairement, il faut en 
    rechercher la racine pour pouvoir trouver delta_t
    '''
    global searchInterval
    x0,x1 = kwargs.get('search_x0',searchInterval[0]),kwargs.get('search_x0',searchInterval[1])
    delta_t ,statut = bissection(f_difference,x0,x1,**kwargs) #delta_t est compris entre 0h et 20h 
    
    if statut !=0 : 
        print('erreur, problème de racine') 
        return (-1, statut)
    return delta_t

def question_4_2(T_max_d,**kwargs):
    '''T_max_d en degrés celsius'''
    global h
    h = kwargs.get('h',h)
    kwargs['h'] = h
    delta_t = recherche_delta_t(T_max_d,**kwargs)
    kwargs['delta_t'] = delta_t
    kwargs['T_max_d'] = T_max_d
    kwargs.pop('num_du_scenario',0)

    if isinstance(delta_t, tuple):
        print("fin avortée")
        return delta_t[1]
    print(f"delta_t correspondant aux critères demandés {delta_t}")
    question_4_1(num_du_scenario = 4 ,**kwargs)
    
#______________________________________________________________________________________________________#
# question 4.3



#EN15251 est une array contenant t0 et tf et Tmin et Tmax -> [8,19,19.5,24]

def verification_EN15251(delta_t,**kwargs):
    global tol_temp
    EN15251 = kwargs.pop('EN15251',np.array([8,19,19.5,24]))
    tol_temp = kwargs.pop('tol_temp', tol_temp)
    kwargs['no_max'] = True

    '''
    ATTENTION ICI tol c'est la tolérance en température pour savoir à quel point c'est convergé
    '''

    t,T_confort  = T_max(delta_t,**kwargs)
    plt.plot(t,T_confort, label = 'température de confort')
    for i in range(2): 
        fonctiondroite(EN15251[i+2], label = ['température minimale','température maximale'][i])
    plt.title(label = "graphique de la température de confort pendant la dernière journée")
    plt.xlabel('heures de la journée (h)')
    plt.ylabel('température (°C)')
    points_to_plot = [('début de la période de chauffe (4h)',4,EN15251[2]),(f'fin de la période de chauffe ({4+delta_t}h)',4+delta_t,EN15251[2]),('début des heures de bureau',EN15251[0],EN15251[2]),('fin des heures de bureau',EN15251[1],EN15251[2])] 
    for i in range(len(points_to_plot)):
        label,x,y = points_to_plot[i]
        plt.plot(x,y,'.', label = label)
    
    
    plt.show()
    for i in range(len(t)):
        T_confort_i = T_confort[i]
        if (EN15251[0]<= t[i]<=  EN15251[1]):            
            if not(EN15251[2]-tol_temp <= T_confort_i <= EN15251[3]+tol_temp): #est ce que les extrémas sont pris avec ?
                print("La norme EN15251 n'est pas respectée.")
                return False
    print("La norme EN15251 est respectée.")
    return True




def max_a_stabilisation(**kwargs): 
    '''
    fonction qui rend le maximum stabilisé au dernier jour
    
    '''
    global T0,FenetreDeTemps,num_du_scenario
    num_du_scenario = kwargs.get('num_du_scenario', num_du_scenario)
    T0 = kwargs.pop('T0',T0)
    kwargs['q_3_5'] = False
    delta_t = kwargs.get('delta_t',0)
    kwargs['delta_t'] = delta_t
    if delta_t != 0 :
        kwargs['num_du_scenario'] = 4
    print (f"on passe au scenario numéro {num_du_scenario}.")
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',FenetreDeTemps)
    #le plot se fait ici
    
    days_to_stabilize, T0_new = cycles_apres_convergence(T0,FenetreDeTemps,**kwargs) #T0_new est les conditions initiales du dernier jour 
    
    kwargs['T0'] = T0_new
    
    
    if days_to_stabilize == None:
        return("erreur de stabilisation")
    delta_t = kwargs.pop('delta_t',0) # cas par défaut for the sake of it
    retour = T_max(delta_t,**kwargs)
    if debug: print(f'maximum: {retour[0]}') 
    return retour

def question_4_3(T_max_d, **kwargs):
    global h, searchInterval,tol_rac,T0,FenetreDeTemps
    kwargs['num_du_scenario'] = 4 
    
    kwargs['h'] = kwargs.get('h',h)
    kwargs['tol_rac'] = kwargs.get('tol_rac',tol_rac)
    kwargs['q_3_5'] = False
    if debug: start=time.time()
    Temp_Max_delta_t =  lambda delta_t: max_a_stabilisation(delta_t,**kwargs)[0] - T_max_d  
    
    x0,x1 = kwargs.get('search_x0',searchInterval[0]),kwargs.get('search_x0',searchInterval[1])
    
    delta_t ,statut = bissection(Temp_Max_delta_t,x0,x1,**kwargs)
    print(f'delta_t : {delta_t}')
    
    if statut:
        print(" il y a eu un souci dans la recherche de racine. On n'a pas trouvé de delta_t qui pouvait atteindre T_max_d")
        return -1
    
    else:
        
        kwargs['delta_t'] = delta_t #mise a jour de delta_t des kwargs
        T0 = kwargs.pop('T0',T0)
        FenetreDeTemps = kwargs.pop('FenetreDeTemps',FenetreDeTemps)
        
        kwargs['T0']= cycles_apres_convergence(T0,FenetreDeTemps,**kwargs)[1] #Mise a jour de T0 pour les conditions du dernier jour
        
        #on doit a nouveau se débarasser de delta_t de kwargs
        delta_t = kwargs.pop('delta_t',0) #fallback on sait jamais
        
        resultat_verif = verification_EN15251(delta_t,**kwargs)
    if debug: 
        end=time.time()
        print(f"Fin des opérations. Temps écoulé: {end-start} secondes")
    return resultat_verif, delta_t
    ''


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

'''    

def plot_T_max_delta_t(**kwargs):
    global searchInterval
    kwargs['num_du_scenario'] = 4
    kwargs['q_3_5'] = False
    T_max_d = kwargs.get('T_max_d',24)
    
    delta_test = np.arange(searchInterval[0],searchInterval[1],0.5)
    T_max_a_delta_t = np.zeros_like(delta_test)
    Temp_Max_delta_t =  lambda delta_t: max_a_stabilisation(delta_t,**kwargs)[0] - T_max_d  
    for i in range(len(delta_test)):
        T_max_a_delta_t[i] = Temp_Max_delta_t(delta_test[i])
    plt.plot(delta_test,T_max_a_delta_t,'.', label="Temp_Max_delta_t")
    plt.xlabel("delta_t")
    plt.ylabel("Temp_Max_delta_t")
    plt.title("Courbe de Temp_Max_delta_t en fonction de delta_t")
    plt.legend()
    plt.grid(True)
    plt.show()
    



