import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
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
        self._tab_df = [None, None, None, None]  # t0, t1, taille, tab_df
        self._tab_angles_R = [None, None, None, None]  # t0, t1, taille, tab_angles_R

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
        axe_x, axe_y = self.plan(t0, t1, taille)
        tab_df = self.tab_df(t0, t1, taille)
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
            print("tab_angles_R true: ", self._tab_angles_R[:3])
            if self._tab_angles_R[3] is not None:
                print("tab_angles_R not None")
                return self._tab_angles_R[3]
        else:
            print("tab_angles_R false, ", self._tab_angles_R[:3])
            self._tab_angles_R[:3] = [t0, t1, taille]
            print("tab_angles_R[:3], ", self._tab_angles_R[:3])

        tab_df = self.tab_df(t0, t1, taille)
        tab_angles_x_2pi = np.arctan2(tab_df[0][0], tab_df[1][0]) % (2 * math.pi)
        tab_angles_y_2pi = np.arctan2(tab_df[0][1], tab_df[1][1]) % (2 * math.pi)

        def corrigeur(tab):
            # 获取列的个数
            # Obtenir le nombre de colonnes
            width = len(tab[0])
            # 声明储存修正后角度的集合
            # Déclarer l'ensemble des angles corrigés
            tab_angle_R = []
            # 按行遍历未修正的角度集合，来修正theta_x的数据
            # Parcourir l'ensemble des angles non-corrigé ligne par ligne, pour corriger les données de theta_x
            for ligne in tab:
                # 声明每行的修正角度集合
                # Déclarer l'ensemble des angles corrigés d'une ligne
                angle_ligne_R = []
                # 初始化每行最开头的角度，奠定随后各点真实角度的偏移基准
                # Initialiser le premier angle d'une ligne. C'est la base de biais de l'angle de point suivant.
                if ligne[0] < math.pi:
                    # 如果每行最初的角度是正方向的（即逆时针）（因为在生成tab_angle_pi时每个角度都被模pi）
                    # 注意ligne中每个元素其实是一个含两个元素的np.array
                    # Si le premier angle est dans le sens trigonométrique (sens inverse (ou contraire) des aiguilles
                    # d'une montre) (car l'angle 'tab_angle_pi' a été modulo Pi après qu'il était généré)
                    # Attention, chaque élément dans 'ligne' est en type de 'np.array' qui contient deux éléments
                    angle_ligne_R.append(ligne[0])
                else:
                    # 反之为逆方向（顺时针）
                    # Sinon, c'est le sens des aiguilles d'une montre
                    angle_ligne_R.append(ligne[0] - 2 * math.pi)

                # 为这一行随后（从左到右）每个角度计算相对于前一个角度的实际偏移量
                # Calculer le biais réel de chaque point suivant par rapport au son point précédent
                for i in range(1, width):
                    # 计算偏移量的绝对值
                    # Calculer la valeur absolue du biais
                    diff = abs(ligne[i] - ligne[i - 1])
                    # 判断偏移是否导致真实值超越arctan的定义域边界
                    # Déterminer si ce biais contuit que la valeur réelle de l'angle passe (la multiplication de) le
                    # bord de l'ensemble de definition de arctan
                    if diff > math.pi:
                        # 如果大于Pi，则说明偏移导致真实角度突破arctan的定义域边界，则真实的偏移角度应该是2*Pi - diff
                        # Si la valeur > Pi, alors le biais constuit le passage, le biais réel doit être 2*Pi - diff
                        diff = 2 * math.pi - diff
                        # 只有两种突破边界的情况：
                        # il n'existe que deux cas du passer le bord:
                        if ligne[i] >= ligne[i - 1]:
                            # 上一个角度大于但接近0，下一个角度（反方向）顺时针偏移后为负角度，取模后变成接近pi的角度
                            # l'angle précédent est proche à 0 (mod 2*Pi), l'angle suivant est dans le sens
                            # trigonométrique, et devient proche à Pi après modulo.
                            angle_ligne_R.append(angle_ligne_R[i - 1] - diff)
                        else:
                            # 上一个角度接近2*pi，下一个角度（正方向）逆时针偏移后超过pi，取模后变成接近0的角度
                            # l'angle précédent est et proche à 2*Pi (mod 2*Pi), l'angle suivant est dans le sens des
                            # aiguilles d'une montre, et devient proche à 0 après modulo.
                            angle_ligne_R.append(angle_ligne_R[i - 1] + diff)
                    else:
                        if ligne[i] >= ligne[i - 1]:
                            # 下一个角度比上一个角度大，（正方向）逆时针偏移
                            # l'angle suivant est plus grand que le précédent, le biais dans le sens trigonométrique
                            angle_ligne_R.append(angle_ligne_R[i - 1] + diff)
                        else:
                            # 下一个角度比上一个角度小，（反方向）顺时针偏移
                            # l'angle suivant est plus petit que le précédent, le biais dans le sens des aiguilles
                            # d'une montre
                            angle_ligne_R.append(angle_ligne_R[i - 1] - diff)
                # 将当前行储存
                # Enregistrer cette ligne courante
                tab_angle_R.append(angle_ligne_R)
            return tab_angle_R

        tab_angles_x_R = corrigeur(tab_angles_x_2pi)
        tab_angles_y_R = corrigeur(tab_angles_y_2pi.T)

        # tab_angles_x_R, tab_angles_y_R=tab_angles_x_2pi,tab_angles_y_2pi.T
        self._tab_angles_R[3] = tab_angles_x_R, tab_angles_y_R
        return self._tab_angles_R[3]


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
le_t0, le_t1, la_taille = -1, 1, 500
ex = fonc_diff_infini(f_ex2(0.2, 5, 5 * math.pi)[0])
### expr = x + 0.45 * sp.exp(-15 * (x ** 2 + y ** 2)), y + 0.2 * sp.exp(-10 * (x ** 2 + y ** 2))
### ex = fonc_diff_infini(expr)
# print(ex.sym())
# print(ex.num())
# print(ex.f(0, 0))
# print(ex.df_sym())
# print(ex.df(0, 0))
ex.draw()
# ex.draw('h')
# ex.draw('v')
# print(ex.tab_df())
# ex.draw_df()
# ex.draw_df('h')
# ex.draw_df('v')
ex.draw_all('h')
ex.draw_all('v')
# print(ex.tab_df(le_t0, le_t0, la_taille))
print(ex.tab_angles_R(-le_t0, le_t1, la_taille))

plt.plot(np.linspace(le_t0, le_t1, la_taille), ex.tab_angles_R(le_t0, le_t1, la_taille)[0][la_taille // 2])
my_y_ticks = np.arange(-math.pi, 1.5 * math.pi, 0.25 * math.pi)
plt.yticks(my_y_ticks)
plt.title("direction x")
plt.xlabel("x")
plt.ylabel('$\Theta$')
plt.show()

plt.plot(np.linspace(le_t0, le_t1, la_taille), ex.tab_angles_R(le_t0, le_t1, la_taille)[1][la_taille // 2])
my_y_ticks = np.arange(-math.pi, 1.5 * math.pi, 0.25 * math.pi)
plt.yticks(my_y_ticks)
plt.title("direction y")
plt.xlabel("y")
plt.ylabel('$\Theta$')
plt.show()
