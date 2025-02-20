from config import *
import SimTABS
import matplotlib.pyplot as plt




#TESTER LES EXERCICES

#t,T = SimTABS.calculCycles(5,T0,FenetreDeTemps,h)

solution = SimTABS.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10)
t = solution.t
T = solution.y

T2 = SimTABS.calculTemperaturesEuler(FenetreDeTemps,T0,0.1, t )

T3 = T - T2[1]
T = T3
if debug: 
    plt.ylabel('Température(T)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.xlabel('Temps (t)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
    for i in range(T.shape[0]):  
        plt.plot(t, T[i], label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
    plt.legend(loc='best')
    plt.show()  

