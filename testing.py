from config import *
import SimTABS
import matplotlib.pyplot as plt

# c'est l'exercice de test qu'on veut appliquer.
# 1 c'est les cycles
#2 c'est la différence (! au custom = True et euler qui fait des siennes )
#3 c'est calcul par euler
#4 par IVP
sim= 4

#TESTER LES EXERCICES
if sim == 1:
    t,T = SimTABS.calculCycles(5,T0,FenetreDeTemps,h)

elif sim ==2:
    t,T = SimTABS.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10)

    t,T2 = SimTABS.calculTemperaturesEuler(FenetreDeTemps,T0,0.001, t )

    T3 = T -T2
    
    T = T2
elif sim == 3:
    t,T = SimTABS.calculTemperaturesEuler(FenetreDeTemps,T0,0.001, 0 )
    
elif sim == 4: 
    t,T = SimTABS.calculTemperaturesIVP(FenetreDeTemps,T0, rtol=10e-10)

if debug: 
    plt.ylabel('Température(T)', fontsize = 8) # Labélisation de l'axe des ordonnées (copypaste du tuto)
    plt.xlabel('Temps (t)', fontsize = 8) # Labélisation de l'axe des abscisses (copypaste du tuto)
    index = ['T_room','T_t','T_cc','T_c1','T_c2','undefined','undefined'] 
    for i in range(T.shape[0]):  
        plt.plot(t, T[i]- 273.15, label=index[i])  # en fonction du nombre de variables dans T, on affiche plus ou moins de fonctions
    plt.legend(loc='best')
    plt.show()  

