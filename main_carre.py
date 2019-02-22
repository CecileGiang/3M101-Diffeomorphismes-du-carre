############################## PROJET 3M101 - DÉFORMATION DE DIFFÉOMORPHISMES DU CARRÉ ##############################

'''
THÉORÈME DE SMALE (1959):
    
    L'espace des difféomorphismes du carré qui sont l'identité
    sur un voisinage du bord est contactile.
    
SIMPLIFICATION:
    
    L'espace métrique des difféomorphimes du carré [0,1]²
    est connexe par arcs.

'''

import math
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

#matplotlib inline
plt.rc('figure', figsize=(12,9))

# Fonction de calcul d'un C_inf-difféormorphisme

'''
On implémente en exemple un difféomorphisme du carré, qui doit être l'identité au bord,
ie que les bords sont invariants par la fonction. Nous la prendrons gaussienne, en 2 dimensions.
Notre fonction f sera telle que f(x,y) = (f1(x,y),f2(x,y)).
'''

def f1(x,y,beta=0.45):
    return x+beta*np.exp(-15*(x**2+y**2))

def f2(x,y,alpha=0.2):
    return y+alpha*np.exp(-10*(x**2+y**2))


def f(x,y,alpha=0.2,beta=0.45):
    '''f sera un C_inf-difféomorphisme de [-1,1]² dans [-1,1]².
       f(x,y) = (f1(x,y),f2(x,y)) avec:
           * f1(x,y) = (x,φ(x,y)), et φ(x,y) = y + αexp(-y²)
           * f2(x,y) = (x,Ψ(x,y)), et Ψ(x,y) = x + βexp(-x²)'''
    return (f1(x,y,alpha),f2(x,y,beta))
       

# Traçage graphique du difféormorphisme

'''Pour rappel on veut un C_inf-difféomorphisme qui est l'identité
sur le carré [0,1]². On vérifie graphiquement que l'on obtient
bien de telles fonctions avec f1 et f2 ainsi définies'''

#CONSTANTES
ALPHA = 4
BETA = 10

I_min = -1 #valeurs d'intervalles
I_max = 1


######################################### TRAÇAGE DE F1 ##################################################################


#Traçage 2D de f1

fig0 = plt.figure()
plt.title("Traçage 2D de f1")
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
xx, yy = np.meshgrid(x, y, sparse=True)
z = f1(xx,yy, ALPHA)
h = plt.contourf(x,y,z)


#Traçage 3D de f1

'''Traçage noir et blanc'''

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)
Z = f1(X, Y, ALPHA)

fig1 = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.contour3D(X, Y, Z, 50, cmap='binary')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Surface f1');

'''Traçage couleur'''
fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax2.set_title('Surface f1');

######################################### TRAÇAGE DE F2 ##################################################################


#Traçage 2D de f2

fig3 = plt.figure()
plt.title("Traçage 2D de f2")
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
xx, yy = np.meshgrid(x, y, sparse=True)
z = f2(xx,yy, BETA)
h = plt.contourf(x,y,z)


#Traçage 3D de f2

'''Traçage noir et blanc'''

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)
Z = f2(X, Y, BETA)

fig4 = plt.figure()
ax4 = plt.axes(projection='3d')
ax4.contour3D(X, Y, Z, 50, cmap='binary')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')
ax4.set_title('Surface f2');

'''Traçage couleur'''
fig5 = plt.figure()
ax5 = plt.axes(projection='3d')
ax5.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax5.set_title('Surface f2');


###################### CONSTRUCTION GRILLE - REPRÉSENTATION DU DIFÉOMORPHISME DU CARRÉ #############################


#Feuillage horizontal: y fixés

Fh = []
eps = 0.05

for y in np.arange(-1,1+eps,eps):
    Fh.append([(x,y) for x in np.arange(-1,1+eps,eps)])


#Feuillage vertical: x fixés

Fv = []

for x in np.arange(-1,1+eps,eps):
    Fv.append([(x,y) for y in np.arange(-1,1+eps,eps)])


#Affichage de la grille unité:

fig1 = plt.figure()
plt.title("GRILLE UNITÉ")

for ligne_h in Fh:
    for ligne_v in Fv:
        plt.plot(ligne_h,ligne_v)


fig2 = plt.figure()
plt.title("GRILLE UNITÉ APRÈS DÉFORMATION PAR LE DIFFÉOMORPHISME")



#Affichage de la grille unité auquel on a appliqué le difféomorphisme:
    

for ligne_h in Fh: #y fixés
    pos_finale = [f(p[0],p[1]) for p in ligne_h]
    
    X = [] #abscisses à plotter
    Y = [] #ordonnées à plotter
    
    for i in range(len(pos_finale)):
        X.append(pos_finale[i][0])
        Y.append(pos_finale[i][1])

    plt.plot(X,Y)

for ligne_v in Fv: #x fixés
    pos_finale = [f(p[0],p[1]) for p in ligne_v]
    
    X = [] #abscisses à plotter
    Y = [] #ordonnées à plotter
    
    for i in range(len(pos_finale)):
        X.append(pos_finale[i][0])
        Y.append(pos_finale[i][1])
    
    plt.plot(X,Y)

############################### VECTEURS ET FONCTIONS D'ANGLE ###################################
    
  
# Calcul du vecteur de l'image d'un point par le difféomorphisme
    
'''
On considère maintenant un point (x,y) auquel on a appliqué notre difféomorphisme.
Le point image de (x,y) est (x',y') (f(x,y) = (x',y')).

On suppose que (x,y) se trouvait sur une ligne horizontale (respectivement verticale).
On essaie maintenant de retrouver l'angle du vecteur passant par ce point par rapport à un axe horizontal (resp. vertical).
Pour cela, on trouve d'abord les coordonnées du vecteur, données par:
    df/dx = (df1/dx(x,y), df/dx(x,y)) (respectivement df/dx = (df1/dy(x,y), df/dy(x,y)))
'''

"""
https://stackoverflow.com/questions/30791504/python-partial-derivatives-easy
"""
