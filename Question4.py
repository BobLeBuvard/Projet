import matplotlib.pyplot as plt
from RechercheRacine import bissection
import numpy as np
from SimTABS import*

#______________________________________________________________________________________________________#
# question 4.1


def T_max(delta_t, **kwargs):
    '''
    Fonction qui calcule le maximum de température de confort d'un cycle (avec un delta T donné).\n
    Fonction à annuler : T_max(delta_t) - T_dmax 

    Paramètres :
    - delta_t : durée de chauffage après 4h. (float64)
    - **kwargs : se référer aux arguments qu'on peut mettre
    
    Retour : 
    - Différence entre Tmax obtenu et Tmax souhaité.
    '''
    # Initialisation des variables
    global gl_h,gl_T0
    kwargs['delta_t'] = delta_t #ajout aux arguments si ce n'est pas déjà le cas
    T0 =kwargs.pop('T0',gl_T0)
    no_max = kwargs.pop('no_max', False)
    h = kwargs.pop('h',gl_h)
    global FenetreDeTemps
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',gl_FenetreDeTemps)
    MAX = 0

    # Calcul
    t, T = calculTemperaturesEuler(FenetreDeTemps, T0,h,**kwargs)
    T_confort = (T[0, :] + T[4, :]) / 2  # T_room = T[0], T_c2 = T[4]
    if no_max: 
        return t,T_confort
    MAX = np.max(T_confort)
    return MAX, t, T_confort #si on ne veut pas de max -: no_max=True ben on ne le calcule pas.


def question_4_1(**kwargs):
    '''fonction qui dessine le max de température de confort en apellant T_max et en plottant les températures adéquates.'''
    # Initialisation des variables
    delta_t = kwargs.pop('delta_t',0)
    T_max_d = kwargs.pop('T_max_d',0)

    # Calcul
    MAX,t,T_confort = T_max(delta_t, **kwargs)
    indice_max = np.where(T_confort  == MAX)[0][0]
    # Récupérer le temps t correspondant à ce maximum
    t_max = t[indice_max]
    t_max_seconde = t_max * 3600

    # Dessin
    print(f"maximum de température de confort: {MAX}°C à t = {t_max_seconde}s, c'est à dire {t_max}h")#print la température max sur l'intervalle en degrés celsius
    plt.xlabel(f"temps ({FenetreDeTemps[1] - FenetreDeTemps[0]}h)") # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.ylabel("température optimale ") # Labélisation de l'axe des abscisses (copypaste du tuto)
    plt.title(label = f'Température de confort sur {FenetreDeTemps[1]-FenetreDeTemps[0]}h : delta_t = {delta_t}')
    plt.plot(t,T_confort ,label= "température de confort")
    plt.axhline(y=T_max_d, color="red", linestyle="--", label="T_max_d") 
    plt.plot(4,T_max_d,'.', label = f'début de la période de chauffe ({4}h)')
    plt.plot(4+delta_t,T_max_d,'.', label = f'fin de la période de chauffe ({4+delta_t}h)')
    plt.legend( loc='best')
    plt.show()
    return MAX
 

#______________________________________________________________________________________________________#
#question 4.2
def recherche_delta_t (T_max_d,**kwargs):
    """
        Fonction qui va rechercher le delta_t tel que l'on ne dépassera jamais T_max_d sur un cycle de 24h\n
        Paramètres:
        - T_max_d (float) : Température maximale désirée en celsius.
        - **kwargs (dict) : se référer aux arguments qu'on peut mettre
        
        Retour :
        - delta_t (float) : Période delta_t nécessaire pour ne pas dépasser T_max_d.
    """

    # Initialisation des variables
    global gl_h, tol_rac, gl_searchInterval
    kwargs.setdefault('h', gl_h)
    kwargs.setdefault('tol_rac', tol_rac)
    x0,x1 = kwargs.get('search_x0',gl_searchInterval[0]),kwargs.get('search_x1',gl_searchInterval[1])
    kwargs.pop('delta_t',None) #on efface le delta_t des kwargs pour la recherche 
    kwargs['T_max_d'] = T_max_d #on ajoute T_max_d aux arguments pour T_max
    kwargs['num_du_scenario'] = 4
    
    # Fonction à annuler
    f_difference = lambda delta_t: T_max(delta_t,**kwargs)[0] - T_max_d 

    # Calcul
    delta_t ,statut = bissection(f_difference,x0,x1,**kwargs) 
    if statut !=0 : 
        print('erreur, problème de racine') 
        return (-1, statut)
    return delta_t

def question_4_2(T_max_d,**kwargs):

    # Initialisation des variables    
    global gl_h
    
    kwargs.setdefault('h', gl_h)
    kwargs['delta_t'] = delta_t
    kwargs['T_max_d'] = T_max_d
    kwargs['num_du_scenario'] = 4

    # Calcul

    delta_t = recherche_delta_t(T_max_d,**kwargs)   
    if isinstance(delta_t, tuple): # c'est un tuple uniquement s'il y a eu une erreur.
        print("fin avortée")
        return delta_t[1]
    print(f"delta_t correspondant aux critères demandés {delta_t}")

    # Dessin (paresseux)
    question_4_1(**kwargs)
    
#______________________________________________________________________________________________________#
# question 4.3



#EN15251 est une array contenant t0 et tf et Tmin et Tmax : [8,19,19.5,24]

def verification_EN15251(delta_t,**kwargs):
    """
    Fonction vérifiant si une norme de conditions de travail est respectée. Par défaut c'est la norme EN15251.
    """
    
    # Initialisation des variables
    global tol_temp, norme
    EN15251 = kwargs.pop('EN15251',norme)
    tol_temp = kwargs.pop('tol_temp', tol_temp)
    kwargs['no_max'] = True
    EstRespecté = True

    # Calcul

    t,T_confort  = T_max(delta_t,**kwargs)
    for i in range(len(t)):
        T_confort_i = T_confort[i]
        if (EN15251[0]<= t[i]<=  EN15251[1]):            
            if not(EN15251[2]-tol_temp <= T_confort_i <= EN15251[3]+tol_temp): #est ce que les extrémas sont pris avec ?
                print("La norme EN15251 n'est pas respectée.")
                EstRespecté = False
    
    print("La norme EN15251 est respectée.")

    # Dessin
    plt.plot(t,T_confort, label = 'température de confort')
    for i in range(2): 
         plt.axhline(y=EN15251[i+2], linestyle="--", label =['température minimale','température maximale'][i], color=["blue", "red"][i])
    plt.title(label = "graphique de la température de confort pendant la dernière journée")
    plt.xlabel('heures de la journée (h)')
    plt.ylabel('température (°C)')
    points_to_plot = [('début de la période de chauffe (4h)',4,EN15251[2]),(f'fin de la période de chauffe ({4+delta_t}h)',4+delta_t,EN15251[2]),('début des heures de bureau',EN15251[0],EN15251[2]),('fin des heures de bureau',EN15251[1],EN15251[2])] 
    for i in range(len(points_to_plot)):
        label,x,y = points_to_plot[i]
        plt.plot(x,y,'.', label = label)
    
    plt.legend(fontsize="5", loc = 'lower right')
    plt.show()
    return EstRespecté




def max_a_stabilisation(delta_t,**kwargs): 
    '''
    fonction qui rend le maximum stabilisé au dernier jour
    
    '''
    global gl_T0,gl_FenetreDeTemps,gl_num_du_scenario   
    kwargs['q_3_5'] = False
    kwargs['delta_t'] = delta_t
    FenetreDeTemps = kwargs.pop('FenetreDeTemps',gl_FenetreDeTemps)
    T0 = kwargs.pop('T0',gl_T0)
    
    days_to_stabilize, T0_new = cycles_apres_convergence(T0,FenetreDeTemps,**kwargs) #T0_new est les conditions initiales du dernier jour + plot
    kwargs['T0'] = T0_new
    
    
    if days_to_stabilize == None:
        return("Erreur de stabilisation.")
    
    delta_t = kwargs.pop('delta_t',0) # cas par défaut for the sake of it
    retour = T_max(delta_t,**kwargs)

    if debug:
        print(f'La température maximal au dernier jour est de : {retour[0]} °C.') 

    return retour

def question_4_3(T_max_d, **kwargs):
    global gl_h, gl_searchInterval,tol_rac,gl_T0,gl_FenetreDeTemps
    kwargs['num_du_scenario'] = 4 
    kwargs['q_3_5'] = False
    kwargs['tol_rac'] = kwargs.get('tol_rac',tol_rac)
    kwargs['h'] = kwargs.get('h',gl_h)
    x0,x1 = kwargs.get('search_x0',gl_searchInterval[0]),kwargs.get('search_x1',gl_searchInterval[1])
   
    Temp_Max_delta_t =  lambda delta_t: max_a_stabilisation(delta_t,**kwargs)[0] - T_max_d  
    
    delta_t ,statut = bissection(Temp_Max_delta_t,x0,x1,**kwargs)
    print(f'delta_t : {delta_t}')
    
    if statut:
        print(" il y a eu un souci dans la recherche de racine. On n'a pas trouvé de delta_t qui pouvait atteindre T_max_d")
        return -1
    
    else:
        
        kwargs['delta_t'] = delta_t #mise a jour de delta_t des kwargs
        T0 = kwargs.pop('T0',gl_T0)
        FenetreDeTemps = kwargs.pop('FenetreDeTemps',gl_FenetreDeTemps)
        
        kwargs['T0']= cycles_apres_convergence(T0,FenetreDeTemps,**kwargs)[1] #Mise a jour de T0 pour les conditions du dernier jour
        
        #on doit a nouveau se débarasser de delta_t de kwargs
        delta_t = kwargs.pop('delta_t',0) #fallback on sait jamais
        
        resultat_verif = verification_EN15251(delta_t,**kwargs)
    if debug: 
        print("Fin des opérations.")
    return resultat_verif, delta_t


#______________________________________________________________________________________________________#
# Partie Bonus

def plot_T_max_delta_t(**kwargs):
    global gl_searchInterval
    kwargs['num_du_scenario'] = 4
    kwargs['q_3_5'] = False
    T_max_d = kwargs.get('T_max_d',24)
    
    delta_test = np.arange(gl_searchInterval[0],gl_searchInterval[1],0.5)
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