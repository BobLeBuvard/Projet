import time
from RechercheRacine import *
sc = [bissection, secante, hybride]
x0 = 0
x1 = 10
def f(x):
    time.sleep(0.1)
    return x**2 -3
for i in range(3):
    start = time.time()
    racine, statut = sc[i](f,x0,x1)
    print(f"racine et statut: {racine,statut}")
    end = time.time()
    print(f"temps {end-start}")