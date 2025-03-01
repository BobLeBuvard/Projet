#DEPRECATED: NE PLUS UTILISER LES FONCTIONS QUI SONT ICI SAUF POUR DES TESTS

from config import *
import SimTABS_old
import matplotlib.pyplot as plt
# sim (simulation) c'est l'exercice de test qu'on veut appliquer.
# 1 c'est les cycles
#2 c'est la différence (! au custom = True et euler qui fait des siennes )
#3 c'est calcul par euler
#4 par IVP

def test(h):
    
    global T1,T2
    t_euler,T2 = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h)
    t,T1 = SimTABS_old.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10,t_eval=t_euler )
    

    T3 = T1 -T2
    if debug: 
        plt.ylabel('Température(T)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
        plt.xlabel('Temps (t)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
        index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
        for i in range(T3.shape[0]):  
            plt.plot(t, T3[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
        plt.legend( loc='best')
        plt.title(label = 'h = '+str(h))
        plt.show()  


sim= 2

#TESTER LES EXERCICES

if sim == 1:
    t,T = SimTABS_old.calculCycles(5,T0,FenetreDeTemps,h)

#question 3.4
elif sim ==2:
    '''tester la différence entre les deux fonctions pour des valeurs de h différentes -> tester la convergence de Euler avec solve_IVP'''
    h_de_test = [0.001, 0.01,0.1,0.25,0.5,1,24]
    for i in range(len(h_de_test)):
        test(h_de_test[i])        
    
elif sim == 3:
    '''calcul de température par Euler'''
    t,T = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h)
    
    

elif sim == 4: 
    '''Calcul par solve_IVP'''
    h = 'méthode de solve_IVP'
    t,T = SimTABS_old.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10)
    

elif sim ==5:
    h = 0.01
    ''' calcul de la différence de température par tranches de 24h --> NECESSITE UN h DIVISEUR DE 24 ( ex: 0.1, 0.01, ou autre )'''
    t,T = SimTABS_old.calculCycles(16,T0,FenetreDeTemps,h)
    T_converge = SimTABS_old.converge(h,T,0.01)
    t2 = t[:len(T_converge)] # les  premiers éléments de t2
    plt.plot (t2,T_converge)
    plt.title(label = 'graphique de la différence de température entre deux jours au cours du temps')
    plt.plot()
    plt.show() 
#main.dessinemoassa(t,T,['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'],ylabel='Température(T)',xlabel ='Temps (t)' ,titre = str(h))

elif sim == 6:
    '''tester la différence entre les résolutions par Euler pour des valeurs de h différentes '''
    h_de_test = [0.001, 0.01,0.1,0.25] #0.001 pas utile puisque 0.01 l'approche suffisemment bien
    for i in range(len(h_de_test) -1 ):
        t_euler2,T2 = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h_de_test[i])   
        t_euler1,T1 = SimTABS_old.calculTemperaturesEuler(FenetreDeTemps,T0,h_de_test[i+1])
        for i in range(5):
            plt.plot(t_euler1,T1[i])
            plt.plot(t_euler2,T2[i])
            plt.show()
    


    
if debug and (sim !=2) and (sim !=6)  : 
    #T = T - 273.15 #remise en celsius
    plt.ylabel('Température(°K)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.xlabel('Temps (heures)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
    for i in range(T.shape[0]):  
        plt.plot(t, T[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
    plt.legend( loc='best')
    plt.title(label = str(h))
    plt.show()  
