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
    df/dx = (df1/dx(x,y), df2/dx(x,y)) (respectivement df/dy = (df1/dy(x,y), df2/dy(x,y)))
'''

"""
https://stackoverflow.com/questions/30791504/python-partial-derivatives-easy
"""

import sympy as sp

"""
######################### NOTE IMPORTANTE #########################

A partir de maintenant on utilise le module sympy. On devra rentrer en argument de fonction
des fonctions sous forme symbolique.

Par exemple pour la première fonction gradient difinie ci-dessous, nous ne pouvons pas
directement l'appliquer aux fonctions f1 et f2 définies plus haut de manière standard.
A la place, il faut les réécrire:
    alpha=0.45
    beta=0.2
    f1=x+alpha*sp.exp(-15*(x**2+y**2))
    f2=y+alpha*np.exp(-10*(x**2+y**2))
"""

def gradient(f1, f2, liste_var='x y'):
    """
    Définition: function*str*var -> list[functions]
    Renvoie le gradient de la fonction f, sous forme d'une liste de fonctions,
    suivant les variables écrites dans liste_var.
    
    gradient(f,'x y')=[df1/dx, df2/dx, df1/dy, df2/dy]
    
    Les variables x et y sont données de la manière suivante: "x y"
    avec des noms de variables sous forme d'une seule lettre
    """
    
    #liste des dérivées partielles 
    gradient=[]

    x,y = sp.symbols(liste_var,real=True)

    gradient.append(sp.lambdify(x,sp.diff(f1,x))) #dérivée partielle de f1 par rapport à x
    gradient.append(sp.lambdify(x,sp.diff(f2,x))) #dérivée partielle de f2 par rapport à x
    gradient.append(sp.lambdify(y,sp.diff(f1,y)))
    gradient.append(sp.lambdify(y,sp.diff(f2,y)))

    return gradient
    
def vecteur_image(f,x,y):
    """
    Définition: function*float*float -> 
    Renvoie les coordonnées du vecteur image de f au point (x,y).
    """
    #caldul pour f1
    
    
    return
