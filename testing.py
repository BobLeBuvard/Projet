from config import *
import SimTABS
import matplotlib.pyplot as plt
# sim (simulation) c'est l'exercice de test qu'on veut appliquer.
# 1 c'est les cycles
#2 c'est la différence (! au custom = True et euler qui fait des siennes )
#3 c'est calcul par euler
#4 par IVP
def test(h):
    
        
    t_euler,T2 = SimTABS.calculTemperaturesEuler(FenetreDeTemps,T0,h)
    t,T1 = SimTABS.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10,t_eval=t_euler )

    

    T3 = T1 -T2
    T = T3
    if debug: 
        plt.ylabel('Température(T)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
        plt.xlabel('Temps (t)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
        index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
        for i in range(T.shape[0]):  
            plt.plot(t, T[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
        plt.legend( loc='best')
        plt.title(label = 'h = '+str(h))
        plt.show()  

sim= 2


#TESTER LES EXERCICES

if sim == 1:
    t,T = SimTABS.calculCycles(5,T0,FenetreDeTemps,h)
    T = T - 273.15

#question 3.4
elif sim ==2:
    #tester la différence entre les deux fonctions pour des valeurs de h différentes
    h_de_test = [0.001, 0.01,0.1,0.25,0.5,1,24]
    for i in range(len(h_de_test)):
        test(h_de_test[i])        
    
elif sim == 3:
    t,T = SimTABS.calculTemperaturesEuler(FenetreDeTemps,T0,h, 0 )
    T = T - 273.15
elif sim == 4: 
    h = 'méthode de solve_IVP'
    t,T = SimTABS.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10)
    T = T - 273.15





if debug & sim !=2 : 
    plt.ylabel('Température(T)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.xlabel('Temps (t)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
    for i in range(T.shape[0]):  
        plt.plot(t, T[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
    plt.legend( loc='best')
    plt.title(label = str(h))
    plt.show()  
