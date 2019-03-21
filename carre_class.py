import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from IPython import display
import scipy
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative


class fonc_diff_infini:
    """
    Classe de fonction en C-diff-infini R^2->R^2
    Ici, on pense que un diffeomorphisme est un objet dans le classe de fonction en C-diff-infini R^2->R^2.
    On declare un tel objet par une expression de sympy, qui prend deux arguments x et y, et retourne deux valeurs.
    Par ex.:
        x,y=sp.symbols("x y")
        expr=(x+y,x*y)
        ex=fonc_diff_infini(expr,(x,y))
    """

    def __init__(self, expr, vars_sym=(sp.symbols("x y"))):
        self._expr = expr
        self._x, self._y = vars_sym
        self._num = sp.lambdify((self._x, self._y), self._expr, "numpy")
        self._df_sym = None
        self._df_num = None
        self._plan = [None, None, None, None]  # t0, t1, taille, plan
        self._tab_f = [None, None, None, None]  # t0, t1, taille, f(plan)
        self._tab_df = [None, None, None, None]  # t0, t1, taille, tab_df
        self._tab_angles_R = [None, None, None, None]  # t0, t1, taille, tab_angles_R
        self._simulation = None

    def sym(self):
        """
        :return: l'expression en sympy du diffeomorphisme
        """
        return self._expr

    def num(self):
        """
        Rrtourner une fonction python de l'expression symbolique du diffeomorphisme
        :return:
        """
        return self._num

    def f(self, x_num, y_num):
        return self._num(x_num, y_num)

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
        if [t0, t1, taille] == self._plan[:3]:
            if self._plan[3] is not None:
                return self._plan[3]
        else:
            self._plan[:3] = [t0, t1, taille]
        axe_x, axe_y = np.meshgrid(np.linspace(t0, t1, taille), np.linspace(t0, t1, taille))
        self._plan[3] = axe_x, axe_y
        return axe_x, axe_y

    def tab_f(self, t0=-1, t1=1, taille=50):
        if [t0, t1, taille] == self._tab_f[:3]:
            if self._tab_f[3] is not None:
                return self._tab_f[3]
        self._tab_f[:3] = [t0, t1, taille]
        axe_x, axe_y = self.plan(t0, t1, taille)
        tab_x, tab_y = self.f(axe_x, axe_y)
        self._tab_f[3] = (tab_x, tab_y)
        return tab_x, tab_y

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
        tab_x, tab_y = self.tab_f(t0, t1, taille)
        plt.plot(tab_x.T, tab_y.T)

    def draw_v(self, t0=-1, t1=1, taille=50):
        tab_x, tab_y = self.tab_f(t0, t1, taille)
        plt.plot(tab_x, tab_y)

    def draw(self, direction='a', t0=-1, t1=1, taille=50):
        """
        Afficher le diffeomorphisme par une image en 2D
        :param direction: soit 'h' pour la direction horientale, soit 'v' pour la direction verticale, soit l'autre pour
         tous afficher en une meme image
        :param t0:
        :param t1:
        :param taille:
        :return:
        """
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
        if [t0, t1, taille] == self._tab_df[:3]:
            if self._tab_df[3] is not None:
                return self._tab_df[3]
        else:
            self._tab_df[:3] = [t0, t1, taille]

        if self._df_num is None:
            self.df_num()

        axe_x, axe_y = self.plan(t0, t1, taille)
        self._tab_df[3] = self._df_num(axe_x, axe_y)
        return self._tab_df[3]

    def draw_df(self, direction='a', t0=-1, t1=1, taille=50):
        """
        Afficher le champ de vecteurs pour un diffeomorphisme, les autres sont parailles que draw
        :param direction:
        :param t0:
        :param t1:
        :param taille:
        :return:
        """
        tab_x,tab_y=self.tab_f(t0, t1, taille)
        tab_df = self.tab_df(t0, t1, taille)
        if direction == 'h':
            plt.quiver(tab_x, tab_y, tab_df[0][0], tab_df[1][0])
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.show()
        elif direction == 'v':
            plt.quiver(tab_x, tab_y, tab_df[0][1], tab_df[1][1])
            plt.xlabel(r'$y_1$')
            plt.ylabel(r'$y_2$')
            plt.show()
        else:
            plt.quiver(tab_x, tab_y, tab_df[0][0], tab_df[1][0])
            plt.quiver(tab_x.T, tab_y.T, tab_df[0][1], tab_df[1][1])
            plt.xlabel(r'$x_1$ et $y_1$')
            plt.ylabel(r'$x_2$ et $y_2$')
            plt.show()

    def draw_all(self, direction='a', t0=-1, t1=1, taille=50):
        """
        Pour un diffeomorphisme, afficher une fois lui-meme et son champ de vecteurs en une figure, les aures sont
        parailles que draw et que draw_df
        :param direction:
        :param t0:
        :param t1:
        :param taille:
        :return:
        """
        if direction == 'h':
            self.draw_h(t0, t1, taille)
        elif direction == 'v':
            self.draw_v(t0, t1, taille)
        else:
            self.draw_h(t0, t1, taille)
            self.draw_v(t0, t1, taille)
        self.draw_df(direction, t0, t1, taille)

    def tab_angles_R(self, t0=-1, t1=1, taille=50):
        if [t0, t1, taille] == self._tab_angles_R[:3]:
            if self._tab_angles_R[3] is not None:
                return self._tab_angles_R[3]
        else:
            self._tab_angles_R[:3] = [t0, t1, taille]

        tab_df = self.tab_df(t0, t1, taille)
        tab_angles_x_2pi = np.arctan2(tab_df[1][0], tab_df[0][0])
        tab_angles_y_2pi = np.arctan2(tab_df[1][1], tab_df[0][1])

        def modulo_2pi(x):
            y = x
            while y > math.pi:
                y -= 2 * math.pi
            while y < -math.pi:
                y += 2 * math.pi
            return y

        def corrigeur(tab):
            tab_R = []
            for ligne in tab:
                ligne_R = [ligne[0]]
                for angle in ligne[1:]:
                    prec = ligne_R[-1]
                    ligne_R.append(prec + modulo_2pi(angle - prec))
                tab_R.append(ligne_R)
            return np.array(tab_R)

        tab_angles_x_R = corrigeur(tab_angles_x_2pi)
        tab_angles_y_R = corrigeur(tab_angles_y_2pi.T)

        self._tab_angles_R[3] = np.array(tab_angles_x_R), np.array(tab_angles_y_R)
        return self._tab_angles_R[3]

    def simulation(self, tab_angles_R, t0=-1, t1=1, taille=50):
        tab_x, tab_y = np.array(tab_angles_R[0]), np.array(tab_angles_R[1])
        pas = (t1 - t0) / (taille - 1)
        axe = np.linspace(t0, t1, taille)

        def tracer(tab_angles):
            tab_points = []
            for i in range(taille):
                ligne_points = [axe[i]]
                for angle in tab_angles[i][:-1]:
                    ligne_points.append(math.tan(angle) * pas)
                tab_points.append(ligne_points)
            return tab_points

        self._simulation = tracer(tab_x), tracer(tab_y.T)
        return self._simulation

    def simulation2(self, t0=-1, t1=1, taille=50):
        angles_x, angles_y = self.tab_angles_R(t0, t1, taille)
        pas = (t1 - t0) / (taille - 1)
        pass

    def draw_sim(self, tab_angles_R, t0=-1, t1=1, taille=50):
        tab_x, tab_y = self.simulation(tab_angles_R, t0, t1, taille)
        axe = np.linspace(t0, t1, taille)
        for i in range(taille):
            plt.plot([axe[i]] * taille, tab_x[i])
        plt.show()


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


""" Zone de tester le code"""
x, y = sp.symbols("x y")
le_t0, le_t1, la_taille = -1, 1, 50
ex = fonc_diff_infini(f_ex2(0.2, 5, 5 * math.pi)[0])
#expr = x + 0.45 * sp.exp(-15 * (x ** 2 + y ** 2)), y + 0.2 * sp.exp(-10 * (x ** 2 + y ** 2))
#ex = fonc_diff_infini(expr)
# print(ex.sym())
# print(ex.num())
# print(ex.f(0, 0))
# print(ex.df_sym())
# print(ex.df(0, 0))
# ex.draw()
# ex.draw('h')
# ex.draw('v')
# ex.draw_new()
# print(ex.tab_df())
# ex.draw_df()
ex.draw_df('h')
ex.draw_df('v')
ex.draw_all('h')
ex.draw_all('v')
# print(ex.tab_df(le_t0, le_t0, la_taille))
# print(ex.tab_angles_R(-le_t0, le_t1, la_taille))
# ex.simulation(ex._tab_angles_R[3], le_t0, le_t1, la_taille)
# ex.draw_sim_h(ex._tab_angles_R[3], le_t0, le_t1, la_taille)

"""
for i in range(la_taille):
    l = np.linspace(le_t0, le_t1, la_taille)
    plt.plot(l, ex.tab_angles_R(le_t0, le_t1, la_taille)[0][i])
    my_y_ticks = np.arange(-math.pi, 2.25 * math.pi, 0.25 * math.pi)
    plt.yticks(my_y_ticks)
plt.title("direction x")
plt.xlabel("x")
plt.ylabel('$\Theta$')
plt.show()
"""

"""
fig  = plt.figure()
nt = 50
l=np.linspace(le_t0, le_t1, la_taille)
for i in range(nt):
    plt.cla()
    t = i*Tmax/nt
    u = u0(x-c*t)
    plt.plot(x,u)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('t =%1.2f' %t)
    display.clear_output(wait=True)
    display.display(plt.gcf())
"""
"""
plt.plot(np.linspace(le_t0, le_t1, la_taille), ex.tab_angles_R(le_t0, le_t1, la_taille)[1][la_taille // 2])
my_y_ticks = np.arange(-math.pi, 2.25 * math.pi, 0.25 * math.pi)
plt.yticks(my_y_ticks)
plt.title("direction y")
plt.xlabel("y")
plt.ylabel('$\Theta$')
plt.show()
"""
