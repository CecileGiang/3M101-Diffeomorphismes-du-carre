#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:08:21 2019

@author: 3530406
"""

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
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#matplotlib inline
plt.rc('figure', figsize=(12,9))

import scipy
from scipy.misc import derivative


######################### On définit les variables x et y dans sympy ###############################################
x, y = sp.symbols("x y")


### Fonctions de définition et de calcul d'un C_inf-difféormorphisme

'''
On implémente en exemple un difféomorphisme du carré, qui doit être l'identité au bord,
ie que les bords sont invariants par la fonction. Nous la prendrons gaussienne, en 2 dimensions.
Notre fonction f sera telle que f(x,y) = (f1(x,y),f2(x,y)).
'''

def define_f(alpha=0.045,beta=0.2):
    """
    Définition: float*float -> python_function*sympy_expression
    Définit f notre C_inf-difféomorphisme de [-1,1]² dans [-1,1]².
       f(x,y) = (f1(x,y),f2(x,y)) avec:
           * f1(x,y) = (x,φ(x,y)), et φ(x,y) = y + αexp(-y²)
           * f2(x,y) = (x,Ψ(x,y)), et Ψ(x,y) = x + βexp(-x²)
    sous deux formes: celle d'une fonction standard python et d'une expression sympy
    """
    #Définit la fonction python f
    def f(x_,y_):
        
        def f1(x_,y_):
            return x_+beta*np.exp(-15*(x_**2+y_**2))
        def f2(x_,y_):
            return y_+alpha*np.exp(-10*(x_**2+y_**2))
        
        return (f1(x_,y_),f2(x_,y_))
    
    #Définit l'expression sympy associée à f
    f_expr=(x+beta*sp.exp(-15*(x**2+y**2)),y+alpha*sp.exp(-10*(x**2+y**2)))
    
    return f,f_expr


def evaluate_f(f_expr,x_,y_):
    """
    Définition: float*float -> (float,float)
    Evalue notre fonction f en (x_,y_) depuis sa forme sympy.
    """
    #f_expr est un tuple. On définit f1 et f2 tels que f_expr=(f1,f2)
    f1, f2 = f_expr

    #On substitue les variables x et y  en x_ et y_ dans les expressions f_1 et f_2
    f1_ = f1.subs([(x,x_),(y,y_)])
    f2_ = f2.subs([(x,x_),(y,y_)])
        
    return (f1_.evalf(),f2_.evalf())


###################### CONSTRUCTION GRILLE - REPRÉSENTATION DU DIFÉOMORPHISME DU CARRÉ #############################


#Feuillage horizontal: y fixés

def feuilletage_h(eps=0.05):
    """
    Retourne le feuilletage horizontal pour x et y variant de -1 à 1, avec un pas eps.
    """
    Fh = []
    
    for y in np.arange(-1,1+eps,eps):
        Fh.append([(x,y) for x in np.arange(-1,1+eps,eps)])
        
    return Fh

#Feuillage vertical: x fixés

def feuilletage_v(eps=0.05):
    """
    Retourne le feuilletage vertical pour x et y variant de -1 à 1, avec un pas eps.
    """
    Fv = []

    for x in np.arange(-1,1+eps,eps):
        Fv.append([(x,y) for y in np.arange(-1,1+eps,eps)])
        
    return Fv


#Affichage de la grille unité:

def grille_unite(eps=0.05,show=True):
    """
    Affiche la grille unité si show=True et renvoie les feuilletages horizontal et vertical, Fh et Fv.
    """
    Fh = feuilletage_h(eps)
    Fv = feuilletage_v(eps)

    if show:
        fig1 = plt.figure()
        plt.title("GRILLE UNITÉ")
        
        for ligne_h in Fh:
            for ligne_v in Fv:
                plt.plot(ligne_h,ligne_v)
        
    return Fh, Fv


#Affichage de la grille unité auquel on a appliqué le difféomorphisme:
    
def grille_diff(f_expr,eps=0.05,show=True):
    """
    Affiche la grille à laquelle on a appliqué notre difféomorphisme f sous la forme d'une expression sympy
    si show=True et renvoie les feuilletages horizontal et vertical, Fh et Fv.
    """
    
    Fh, Fv = grille_unite(show=False)
    
    Fh_diff=Fh.copy()
    Fv_diff=Fv.copy()
    
    for ligne_h in Fh_diff: #y fixés
        ligne_h = [evaluate_f(f_expr,p[0],p[1]) for p in ligne_h]        
        
        if show:
            X = [] #abscisses à plotter
            Y = [] #ordonnées à plotter
            
            for i in range(len(ligne_h)):
                X.append(ligne_h[i][0])
                Y.append(ligne_h[i][1])
            
            plt.plot(X,Y)
    
    for ligne_v in Fv: #x fixés
        ligne_v = [evaluate_f(f_expr,p[0],p[1]) for p in ligne_v]
        
        if show:
            X = [] #abscisses à plotter
            Y = [] #ordonnées à plotter
            
            for i in range(len(ligne_v)):
                X.append(ligne_v[i][0])
                Y.append(ligne_v[i][1])
            
            plt.plot(X,Y)
            
    return Fh_diff, Fv_diff


"""
Test:

f_expr=define_f()[1]
grille_diff(f_expr)

"""


############################### VECTEURS ET FONCTIONS D'ANGLE ###################################
    
    
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

# Calculs différentiels

def jacobienne_expr(f_):
    """
    Définition: f：[-1,1]²->[-1,1]² -> J(f) = | df1/dx  df1/dy |
                                              | df2/dx  df2/dy |
    Pour une fonction symbolique f, on calcule sa matrice différentielle (ie. sa Jacobienne).
    Le résultat est une liste contenant les éléments de J(f) de haut en bas, de gauche à droite.
    """
    return sp.Matrix([[sp.diff(f_[0], x), sp.diff(f_[0], y)], [sp.diff(f_[1], x), sp.diff(f_[1], y)]])


def jacobienne_num(f_,x_,y_):
    """
    Définition; f：[-1,1]²->[-1,1]² -> J(f) = | df1/dx  df1/dy |
                                              | df2/dx  df2/dy |
    Pour une fonction symbolique f, on évalue sa matrice différentielle (ie. sa Jacobienne) en les points x_ et y_
    Le résultat est une liste contenant les éléments de J(f) de haut en bas, de gauche à droite.
    """
    J=jacobienne_expr(f_)
    
    j1= J[0].subs([(x,x_),(y,y_)]).evalf() #df1/dx
    j2= J[1].subs([(x,x_),(y,y_)]).evalf() #df1/dy
    j3= J[2].subs([(x,x_),(y,y_)]).evalf() #df2/dx
    j4= J[3].subs([(x,x_),(y,y_)]).evalf() #df2/dy
    
    return np.array([[j1,j2],[j3,j4]])


# Calcul du vecteur de l'image d'un point par le difféomorphisme
  
def vecteur_xy_h(f_, x_, y_): 
    """
    Retourne le vecteur horizontal correspondant au difféomorphisme f_ (f_ une expression sympy)
    pris au point (x_,y_)
    """
    J=jacobienne_num(f_,x_,y_)
    return (J[0][0],J[1][0])

def vecteur_xy_v(f_, x_, y_): 
    """
    Retourne le vecteur vertical correspondant au difféomorphisme f_ (f_ une expression sympy)
    pris au point (x_,y_)
    """
    J=jacobienne_num(f_,x_,y_)
    return (J[0][1],J[1][1])


# Calcul du champ de vecteur d'un difféomorphisme sur une grille unité

def champ_vecteur(f_):
    """
    Retourne le champ de vecteur du difféomorphisme f sous forme d'expression sympy appliqué à une grille unité.
    En chaque point f(x,y), on calcule les vecteurs vh et vv tels que:
        vh = | df1/dx |         vv = | df1/dy |
             | df2/dx |              | df2/dy |
    """
    Fh, Fv = grille_unite(show=False)
 
    #Calcul du champ de vecteurs pour Fh
    Vh=[]
    
    for ligne_h in Fh:
        vh=[[(jacobienne_num(f_,p[0],p[1])[0][0],jacobienne_num(f_,p[0],p[1])[1][0]),(jacobienne_num(f_,p[0],p[1])[0][1],jacobienne_num(f_,p[0],p[1])[1][1])] for p in ligne_h]
        Vh.append(vh)
    
    
    #Calcul du champ de vecteurs pour Fv
    Vv=[]
    
    for ligne_v in Fv:
        vv=[[(jacobienne_num(f_,p[0],p[1])[0][0],jacobienne_num(f_,p[0],p[1])[1][0]),(jacobienne_num(f_,p[0],p[1])[0][1],jacobienne_num(f_,p[0],p[1])[1][1])] for p in ligne_v]
        Vv.append(vv)
    
    return Vh, Vv


def champ_vecteur_bis(f_):
    """
    Retourne le champ de vecteur du difféomorphisme f sous forme d'expression sympy appliqué à une grille unité.
    En chaque point f(x,y), on calcule les vecteurs vh et vv tels que:
        vh = | df1/dx |         vv = | df1/dy |
             | df2/dx |              | df2/dy |
    Ainsi on retourne une liste de liste (pour chaque ligne de feuilletage) de couples de couples vecteurs:
        f(x,y) -> [(vh1,vh2),(vv1,vv2)]
    """
    Fh, Fv = grille_unite(show=False)
 
    #Calcul du champ de vecteurs pour Fh
    Vh=[]
    
    for ligne_h in Fh:
        vh=[vecteur_xy_h(f_,p[0],p[1]) for p in ligne_h]
        Vh.append(vh)
    
    
    #Calcul du champ de vecteurs pour Fv
    Vv=[]
    
    for ligne_v in Fv:
        vv=[vecteur_xy_v(f_,p[0],p[1]) for p in ligne_v]
        Vv.append(vv)
    
    return Vh, Vv



"""     TEMPS D'EXÉCUTION POUR LES CALCULS DE CHAMPS DE VECTEURS

import time

#Difféomorphisme sous forme d'expression sympy
f_expr=define_f()[1]

#champ vecteur
start=time.time()
champ_vecteur(f_expr)
end=time.time()
print("Temps d'exécution pour champ_vecteur:",end-start)

#champ vecteur bis
start=time.time()
champ_vecteur_bis(f_expr)
end=time.time()
print("Temps d'exécution pour champ_vecteur bis:",end-start)

#Temps d'exécution pour champ_vecteur: 8.246649265289307
#Temps d'exécution pour champ_vecteur bis: 5.963469982147217
"""
"""
def angle(Vh,Vv,eps=0.00000001):

    #Retourne les listes de couples d'angles réels Ah et Av correspondant respectivement au champ de vecteur composé de Vh
    #(vecteurs pour Fh) et Vv (vecteurs pour Fv).

    Ah=[]
    Av=[]
    #Calcul d'angles pour les vecteurs de Vh -feuilletage horizontal
    for ligne_h in Vh:
        v0=ligne_h[0]
        ah=[(math.atan2(v0[0][0],v0[0][1]),math.atan2(v0[1][0],v0[1][1]))] #angle du 1er vecteur de la ligne
        for v in ligne_h[1:]:
            angle_h=math.atan2(v[0][0],v[0][1])
            angle_v=math.atan2(v[1][0],v[1][1])
      
            #À RAJOUTER: rendre les angles réels
          
            ah.append((angle_h,angle_v))
        Ah.append(ah)
    
    #Calcul d'angles pour les vecteurs de Vv -feuilletage vertical
    
    for ligne_v in Vv:
        v0=ligne_v[0]
        av=[(math.atan2(v0[0][0],v0[0][1]),math.atan2(v0[1][0],v0[1][1]))] #angle du 1er vecteur de la ligne
        for v in ligne_v[1:]:
            angle_h=math.atan2(v[0][0],v[0][1])
            angle_v=math.atan2(v[1][0],v[1][1])
     
            #À RAJOUTER: rendre les angles réels
        
            av.append((angle_h,angle_v))
        Av.append(av)
    
    Ah=np.asarray(Ah)
    Av=np.array(Av)
    
    return Ah,Av
"""
"""
def angle_bis(Vh,Vv):
#    Retourne la liste des angles horizontaux pour Vh et des angles verticaux pour Vv,
#    avec Vh et Vv respectivement la liste des vecteurs horizontaux de Fh
#    et la liste des vecteurs horizontaux de Fv.
    Ah=[]
    Av=[]
    #Calcul d'angles pour les vecteurs de Vh -feuilletage horizontal
    for ligne_h in Vh:from s ci p y . i n t e g r a t e import o d ei n t
        ah=[(math.atan2(ligne_h[0][0],ligne_h[0][1]))] #angle du 1er vecteur de la ligne
        for v in ligne_h[1:]:
            angle_h=math.atan2(v[0],v[1])
            #À RAJOUTER: rendre les angles réels
            
            
            
          
            ah.append(angle_h)
        Ah.append(ah)
    
    #Calcul d'angles pour les vecteurs de Vv -feuilletage vertical
    
    for ligne_v in Vv:
        av=[(math.atan2(ligne_v[0][0],ligne_v[0][1]))] #angle du 1er vecteur de la ligne
        for v in ligne_v[1:]:
            angle_v=math.atan2(v[0],v[1])
            
            #À RAJOUTER: rendre les angles réels
          
            av.append(angle_v)
        Av.append(av)
        
    Ah=np.asarray(Ah)
    Av=np.array(Av)
    
    return Ah,Av
"""

def angle_bis(Vh,Vv):
#    Retourne la liste des angles horizontaux pour Vh et des angles verticaux pour Vv,
#    avec Vh et Vv respectivement la liste des vecteurs horizontaux de Fh
#    et la liste des vecteurs horizontaux de Fv.
    Ah=[]
    Av=[]
    #Calcul d'angles pour les vecteurs de Vh -feuilletage horizontal
    """
    for ligne_h in Vh:
        ah=[math.atan2(ligne_h[0][0],ligne_h[0][1])%(2*math.pi)] #angle du 1er vecteur de la ligne
        for v in ligne_h[1:]:
            angle_h=math.atan2(v[0],v[1])%(2*math.pi)
            #À RAJOUTER: rendre les angles réels
            diff=math.fabs(angle_h-ah[-1])
            if diff>math.pi:
                diff=2*math.pi-diff
                if angle_h>=ah[-1]:
                    ah.append(ah[-1]-diff)
                else:
                    ah.append(ah[-1]+diff)
            else:
                if angle_h>=ah[-1]:
                    ah.append(ah[-1]+diff)
                else:
                    ah.append(ah[-1]-diff)
                    
        Ah.append(ah)
        
    """
    
    """
    for ligne_h in Vh:
        ah=[]
        #test sur le 1er angles de la ligne
        a0=math.atan2(ligne_h[0][0],ligne_h[0][1])%(2*math.pi) #angle du 1er vecteur de la ligne modulo 2*pi
        if a0 < math.pi:
            ah.append(a0)
        else:
            ah.append(a0-2*math.pi)
        #Calcul des angles pour le reste des vecteurs de la ligne
        for v in ligne_h[1:]:
            angle_h=math.atan2(v[0],v[1]) #angle brut calculé avec atan2
            angle_h_modulo=angle_h%(2*math.pi) #angle modulo 2*pi qui servira pour calculer la différence avec l'angle précédent
            #À RAJOUTER: rendre les angles réels
            diff=abs(angle_h_modulo-ah[-1]%(2*math.pi)) #différence entre les deux angles modulo 2*pi
            if diff>math.pi:
                diff=2*math.pi-diff 
                if angle_h_modulo>=ah[-1]%(2*math.pi):
                    ah.append(ah[-1]-diff)
                else:
                    ah.append(ah[-1]+diff)
            else:
                if angle_h_modulo>=ah[-1]%(2*math.pi):
                    ah.append(ah[-1]+diff)
                else:
                    ah.append(ah[-1]-diff)
                    
        Ah.append(ah)
    """
    
    def modulo_2pi(angle):
        new_angle=angle
        while new_angle>math.pi:
            new_angle -= 2*math.pi
        while new_angle < -math.pi:
            new_angle += 2*math.pi
        return new_angle
    
    #Calcul d'angles pour les vecteurs de Vh -feuilletage horizontal
    for ligne_h in Vh:
        #ah=[]
        #test sur le 1er angles de la ligne
        a0=math.atan2(ligne_h[0][1],ligne_h[0][0]) #angle du 1er vecteur de la ligne modulo 2*pi
        ah=[a0]
        #Calcul des angles pour le reste des vecteurs de la ligne
        for v in ligne_h[1:]:
            angle_h=math.atan2(v[1],v[0]) #angle brut calculé avec atan2
            #À RAJOUTER: rendre les angles réels
            diff=modulo_2pi(angle_h-ah[-1]) #différence entre les deux angles modulo 2*pi
            ah.append(ah[-1]+diff)
        Ah.append(ah)
        
    #Calcul d'angles pour les vecteurs de Vv -feuilletage vertical
    for ligne_v in Vv:
        #av=[]
        #test sur le 1er angles de la ligne
        a0=math.atan2(ligne_v[0][1],ligne_v[0][0]) #angle du 1er vecteur de la ligne modulo 2*pi
        av=[a0]
        #Calcul des angles pour le reste des vecteurs de la ligne
        for v in ligne_v[1:]:
            angle_v=math.atan2(v[1],v[0]) #angle brut calculé avec atan2
            #À RAJOUTER: rendre les angles réels
            diff=modulo_2pi(angle_v-av[-1]) #différence entre les deux angles modulo 2*pi
            av.append(av[-1]+diff)
        Av.append(av)
        
    Ah=np.asarray(Ah)
    Av=np.array(Av)

    return Ah,Av


"""
########################################## PROBLÈME POUR CETTE FONCTION ##########################################
"""

def plot_from_angles_Av(Av,eps=0.05):
    X=np.arange(-1,1+eps,eps)
    for i in range(len(X)):
        av=Av[i]
        Y=[X[i]+math.tan(av[0]*eps)]
        for angle in av[1:]:
            Y.append(Y[-1]+math.tan(angle)*eps)
        plt.plot(X,Y)
    return

from scipy.integrate import odeint

"""
def integrate(vv,eps=0.05):
    tps=np.linspace(-1,1+eps,eps)
    y_init=vv[0]
    y=odeint(vv[1:],y_init,tps)
    plt.figure()
    plt.plot(tps,y[:,0])
     
    
    return
"""




def euler(f_expr,df): #df le pas
    """
    Définition: Implémente la méthode d'Euler pour intégrer notre champ de vecteurs.
    """
    
    #Construction de la grille du difféomorphisme
    Fh,Fv = grille_diff(f_expr)
    
    #Construction du champ de vecteurs associé
    Vh,Vv = champ_vecteur_bis(f_expr)
    
    #Construction de la liste des angles horizontal et vertocal pour chaque point
    Ah,Av = angle_bis(Vh,Vv)
    
    #Chaque point de la grille apparaît à la fois dans le feuilletage horizontal et vertical
    #On se fixe maintenant au feuilletage horizontal
    X=[]
    for i in range(len(Fh)):
        ligne_p=Fh[i]
        ligne_X=[ligne_p[0]]
        for j in range(len(ligne_p[1:])):
            x,y = ligne_p[0]
            ligne_X.append(ligne_X[-1]+df*np.array([math.cos(Ah[i][j]),math.sin(Ah[i][j])]))
        X.append(ligne_X)
    
    return X
