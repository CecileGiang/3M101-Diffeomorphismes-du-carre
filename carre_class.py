import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative


class fonc_diff_infini:
    'Classe de fonction en C-diff-infini R^2->R^2'

    def __init__(self, expr, vars_sym=(sp.symbols("x y"))):
        self._expr = expr
        self._x, self._y = vars_sym
        self._num = sp.lambdify((self._x, self._y), self._expr, "numpy")
        self._df_sym = None
        self._df_num = None
        self.t0, self.t1, self.taille = None, None, None # pas encore utilises
        self._tab_df = None # pas encore utilises
        self._tab_angles = None # pas encore utilises

    def sym(self):
        return self._expr

    def num(self):
        """
        Rrtourner une fonction python de l'expression symbolique du diffeomorphisme
        :return:
        """
        return self._num

    def f(self, x_num, y_num):
        return self._num(x_num, y_num)

    def df_sym(self):
        """
        对一个符号表达函数f：I^2->I^2求其微分表达式。注意返回的矩阵是一个一维列表，先从上到下，再从左到右
        Pour une fonction symbolique f：I^2->I^2, on calcul son différentiel.
        Attention, le résultat est un liste en 1 dimensionla, en lisant la matrice de différentiel de haut à bas,
        et de gauche à droit
        :param _f:
        :return: | &φ1/&x  &φ1/&y |
                  | &φ2/&x  &φ2/&y |
        """
        if self._df_sym is None:
            self._df_sym = sp.Matrix([[sp.diff(self._expr[0], self._x), sp.diff(self._expr[0], self._y)],
                                      [sp.diff(self._expr[1], self._x), sp.diff(self._expr[1], self._y)]])
        return self._df_sym

    def df_num(self):
        """
        Rrtourner une fonction python de l'expression symbolique du differentiel du diffeomorphisme
        :return:
        """
        if self._df_num is None:
            self._df_num = sp.lambdify((self._x, self._y), self.df_sym(), "numpy")
        return self._df_num

    def df(self, x_num, y_num):
        """
        C'est la version de fonction python du differentiel du diffeomorphisme, qui prend en argument x_num et y_num,
        et qui retourne le differentiel (matrice jacobienne) dans ce point.
        :param x_num:
        :param y_num:
        :return:
        """
        return self.df_num()(x_num, y_num)

    def draw_h(self, t0=-1, t1=1, taille=50):
        I = np.linspace(t0, t1, taille)
        for y_ in np.linspace(t0, t1, taille):  # y fixé
            pos_init = [(x_, y_) for x_ in I]  # ligne horizontale d'ordonnée y dans le carré unité
            pos_finale = [self.f(p[0], p[1]) for p in pos_init]
            X = []  # abscisses à plotter
            Y = []  # ordonnées à plotter
            for i in range(len(pos_finale)):
                X.append(pos_finale[i][0])
                Y.append(pos_finale[i][1])
            plt.plot(X, Y)

    def draw_v(self, t0=-1, t1=1, taille=50):
        I = np.linspace(t0, t1, taille)
        for x_ in np.linspace(t0, t1, taille):  # x fixé
            pos_init = [(x_, y_) for y_ in I]
            pos_finale = [self.f(p[0], p[1]) for p in pos_init]  # juste une translation de la même courbe
            X = []
            Y = []
            for i in range(len(pos_finale)):
                X.append(pos_finale[i][0])
                Y.append(pos_finale[i][1])
            plt.plot(X, Y)

    def draw(self, direction='a', t0=-1, t1=1, taille=50):
        if direction == 'h':
            self.draw_h(t0, t1, taille)
            plt.show()
        elif direction == 'v':
            self.draw_v(t0, t1, taille)
            plt.show()
        else:
            self.draw_h(t0, t1, taille)
            self.draw_v(t0, t1, taille)
            plt.show()

    def plan(self, t0=-1, t1=1, taille=50):
        """
        Retourner deux tableaux qui sont les 'rastérisations' (feuillages) d'un plan traitees par numpy.meshgrid.
        Attention, la structure de ces deux tableaux sont specifiques. Veuilliez-vous afficher ces deux tableaux pour
        la connaitre. C'est pour faciliter le calcul d'apres.
        :param t0:
        :param t1:
        :param taille:
        :return:
        """
        axe_x, axe_y = np.meshgrid(np.linspace(t0, t1, taille), np.linspace(t0, t1, taille))
        return axe_x, axe_y

    def tab_df(self, t0=-1, t1=1, taille=50):
        """
        En utilisant un plan feuillage par numpy.meshgrid, on symplifie le code pour calculer un tableau de matrice
        jacobienne de chaque point dans le plan.
        Attention, le resultat retourne est en la structure de meshgrid. Voir detailles dans la partie ":return"
        :param t0:
        :param t1:
        :param taille:
        :return: le resultat est en numpy.array. Soit [[a,b],[c,d]] la matrice jacobienne du diffeomorphisme dans
        un point, resultat[0][0] contient les a de chaque point ET DANS LA TRUCTURE DE numpy.meshgrid. De meme, 
        resultat[0][1] contient les c, resultat[1][0] contient les b, et resultat[1][1] contient les d.
        """
        if self._df_num is None:
            self.df_num()
        axe_x, axe_y = self.plan(t0, t1, taille)
        return self.df(axe_x, axe_y)

    def draw_df(self, direction='a', t0=-1, t1=1, taille=50):
        tab_df = self.tab_df(t0, t1, taille)
        axe_x, axe_y = self.plan(t0, t1, taille)
        if direction == 'h':
            plt.quiver(axe_x, axe_y, tab_df[0][0], tab_df[1][0])
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.show()
        elif direction == 'v':
            plt.quiver(axe_x, axe_y, tab_df[0][1], tab_df[1][1])
            plt.xlabel(r'$y_1$')
            plt.ylabel(r'$y_2$')
            plt.show()
        else:
            plt.quiver(axe_x, axe_y, tab_df[0][0], tab_df[1][0])
            plt.quiver(axe_x, axe_y, tab_df[0][1], tab_df[1][1])
            plt.xlabel(r'$x_1$ et $y_1$')
            plt.ylabel(r'$x_2$ et $y_2$')
            plt.show()

    def draw_all(self):
        pass

    def tab_angles_R(self, t0=-1, t1=1, taille=50):
        pass


def f_ex(a, b, x_sym=sp.Symbol('x'), y_sym=sp.Symbol('y')):
    """
    返回同一个函数的两个形式，第一个用sympy符号表达，第二个用python函数表达
    Retouner deux formes d'une même fonction mathematique. La première est exprimée par des symboles de sympy,
    la deuxième est exprimée par une fonction de python.
    :param a:
    :param b:
    :param x_sym:
    :param y_sym:
    :return:
    """

    def f_num(x_num, y_num):
        return x_num, y_num + a * np.exp(-b * y_num ** 2 - b * x_num ** 2)

    return (x_sym, y_sym + a * sp.exp(-b * y_sym ** 2 - b * x_sym ** 2)), f_num


def g_ex2(a, b, x_sym=sp.Symbol('x'), y_sym=sp.Symbol('y')):
    g_sym = a * sp.exp(-b * (x_sym ** 2 + y_sym ** 2))

    def g_num(x_num, y_num):
        return a * np.exp(-b * (x_num ** 2 + y_num ** 2))

    return g_sym, g_num


def r_ex2(_theta, g_sym, g_num, x_sym=sp.Symbol('x'), y_sym=sp.Symbol('y')):
    f_new_sym = (sp.cos(_theta * g_sym) * x_sym - sp.sin(_theta * g_sym) * y_sym,
                 sp.sin(_theta * g_sym) * x_sym + sp.cos(_theta * g_sym) * y_sym)

    def f_new_num(x_num, y_num):
        temps = g_num(x_num, y_num)
        return np.cos(_theta * temps) * x_num - np.sin(_theta * temps) * y_num, np.sin(_theta * temps) * x_num + np.cos(
            _theta * temps) * y_num

    return f_new_sym, f_new_num


def f_ex2(a, b, _theta, x_sym=sp.Symbol('x'), y_sym=sp.Symbol('y')):
    g_sym, g_num = g_ex2(a, b, x_sym, y_sym)
    return r_ex2(_theta, g_sym, g_num, x_sym, y_sym)


ex = fonc_diff_infini(f_ex2(0.2, 5, 5 * math.pi)[0])
print(ex.sym())
print(ex.num())
print(ex.f(0, 0))
print(ex.df_sym())
print(ex.df(0, 0))
print(ex.tab_df())
ex.draw()
ex.draw('h')
ex.draw('v')
ex.draw_df()
ex.draw_df('h')
ex.draw_df('v')
