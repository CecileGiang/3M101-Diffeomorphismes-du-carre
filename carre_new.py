import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import scipy
from scipy.misc import derivative

# variables
x, y = sp.symbols("x y")


def f_ex(a, b):
    """
    返回同一个函数的两个形式，第一个用sympy符号表达，第二个用python函数表达
    Retouner deux formes d'une même fonction mathematique. La première est exprimée par des symboles de sympy,
    la deuxième est exprimée par une fonction de python.
    :param a:
    :param b:
    :return:
    """

    def f_num(_x, _y):
        # return _x + a * math.exp(-b * _x ** 2), _y + a * math.exp(-b * _y ** 2)
        # return _x, _y + a * math.exp(-b * _y ** 2 - b * _x ** 2)
        return _x, _y + a * math.exp(-b * _y ** 2 - b * _x ** 2)  # para de l'image reussi = 0.2, 10

    return (x, y + a * sp.exp(-b * y ** 2 - b * x ** 2)), f_num


def f_ex2(a, b):
    def f_num(_x, _y):
        return a * math.exp(-b * (_x ** 2 + _y ** 2))

    f_sym = a * sp.exp(-b * (x ** 2 + y ** 2))
    return f_sym, f_num


def r_ex(_theta, _f_sym, _f_num):
    # mat = np.array([[math.cos(_theta), -math.sin(_theta)], [math.cos(_theta), math.cos(_theta)]])

    f_new_sym = (sp.cos(_f_sym) * x - sp.sin(_f_sym) * y, sp.sin(_f_sym) * x + sp.cos(_f_sym) * y)

    def f_new_num(_x, _y):
        # temps = np.array(_f_num(_x, _y))
        temps = _f_num(_x, _y)
        # return np.dot(mat, temps)
        return math.cos(_theta * temps) * _x - math.sin(_theta * temps) * _y, math.sin(_theta * temps) * _x + math.cos(
            _theta * temps) * _y

    return f_new_sym, f_new_num
    # return f_new_num


def diff_sym(_f):
    """
    对一个符号表达函数f：I^2->I^2求其微分表达式。注意返回的矩阵是一个一维列表，先从上到下，再从左到右
    Pour une fonction symbolique f：I^2->I^2, on calcul son différentiel.
    Attention, le résultat est un liste en 1 dimensionla, en lisant la matrice de différentiel de haut à bas,
    et de gauche à droit
    :param _f:
    :return: | &φ1/&x  &φ1/&y |
              | &φ2/&x  &φ2/&y |
    """
    return sp.Matrix([[sp.diff(_f[0], x), sp.diff(_f[0], y)], [sp.diff(_f[1], x), sp.diff(_f[1], y)]])


def diff_num(_df_sym, _point):
    """
    对给出的一个2*2的微分表达式，和某一个给定的点（a，b）进行数值替换（x<-a, y<-b）
    Etant donné une expression de différentiel (matrice 2*2) et un point particulier (a,b), on substitue x et y
    resp. par a et b
    :param _f_sym:
    :param _point:
    :return: 返回的表格是以“行”为单位的，二维表格
                le tableau en 2 dimensions retourné est basé sur lignes.
    """
    x1 = _df_sym[0].subs({x: _point[0], y: _point[1]})
    y1 = _df_sym[1].subs({x: _point[0], y: _point[1]})
    x2 = _df_sym[2].subs({x: _point[0], y: _point[1]})
    y2 = _df_sym[3].subs({x: _point[0], y: _point[1]})

    mat = np.array([[x1, y1], [x2, y2]])
    # return normer(mat)
    return mat


def angle_num(_df_num):
    """
    接收一个2*2的矩阵，返回其每列对应的角度，每个角的值域为[-pi/2,pi/2]
    Etant donné une matrice 2*2, retourne ses angles dont valeurs sont entre -pi/2 et pi/2 par rapport à ses colonnes.
    :param _df_num:
    :return:
    """
    a1, a2 = 0, 0
    if _df_num[1][0] == 0:
        if _df_num[0][0] >= 0:
            a1 = math.inf
        else:
            a1 = -math.inf
    else:
        a1 = _df_num[0][0] / _df_num[1][0]
    theta_h = math.atan(a1)

    if _df_num[1][1] == 0:
        if _df_num[0][1] >= 0:
            a2 = math.inf
        else:
            a2 = -math.inf
    else:
        a2 = _df_num[0][1] / _df_num[1][1]
    theta_v = math.atan(a2)
    return np.array([theta_h, theta_v])


def tab_angle_num(_tab_carre_point, _df_sym):
    """
    对给出的平面点集和向量场，求每一点的向量角，其中向量角的值域为R，从而使角度值的变化是连续的。
    注意需要相邻点的间距足够小
    Etant donné un tableau des point du plan et un champ de vectuer(en symboles), on calcul, pour chaque point,
    ses angles des vecteurs, où l'ensemble d'arriver de valeur des angles est R, ce qui est pour que le changement de
    valeur d'angle est continue.
    Attention, la différence des points adjacents doit être suffisamment petite.
    :param _tab_carre_point: ex. [[(-1,1) (0,1) (1,1)],
                                  [(-1,0) (0,0) (1,0)],
                                  [(-1,-1) (0,-1) (1,-1)]]
    :param _df_sym:
    :return: [(theta_h(a),theta_v(a)) for a in _tab_carre_point]
    """

    """获取各点的角度mod Pi
        Calculer les angles de chaque point mod Pi"""
    tab_angle_pi = []
    for ligne in _tab_carre_point:
        angle_ligne = []
        for _point in ligne:
            df_point_num = diff_num(_df_sym, _point)
            angle_point_num = angle_num(df_point_num)
            angle_ligne.append(angle_point_num % math.pi)
        tab_angle_pi.append(angle_ligne)

    """修正各点的角度R  x方向
        Corriger ses angles dont valeurs en R pour chaque point (la direction x)"""
    # 获取列的个数
    # Obtenir le nombre de colonnes
    width = len(tab_angle_pi[0])
    # 声明储存修正后角度的集合
    # Déclarer l'ensemble des angles corrigés
    tab_angle_R = []
    # 按行遍历未修正的角度集合，来修正theta_x的数据
    # Parcourir l'ensemble des angles non-corrigé ligne par ligne, pour corriger les données de theta_x
    for ligne in tab_angle_pi:
        # 声明每行的修正角度集合
        # Déclarer l'ensemble des angles corrigés d'une ligne
        angle_ligne_R = []
        # 初始化每行最开头的角度，奠定随后各点真实角度的偏移基准
        # Initialiser le premier angle d'une ligne. C'est la base de biais de l'angle de point suivant.
        if ligne[0][0] < math.pi / 2:
            # 如果每行最初的角度是正方向的（即逆时针）（因为在生成tab_angle_pi时每个角度都被模pi）
            # 注意ligne中每个元素其实是一个含两个元素的np.array
            # Si le premier angle est dans le sens trigonométrique (sens inverse (ou contraire) des aiguilles d'une
            # montre) (car l'angle 'tab_angle_pi' a été modulo Pi après qu'il était généré)
            # Attention, chaque élément dans 'ligne' est en type de 'np.array' qui contient deux éléments
            angle_ligne_R.append([ligne[0][0], None])
        else:
            # 反之为逆方向（顺时针）
            # Sinon, c'est le sens des aiguilles d'une montre
            angle_ligne_R.append([ligne[0][0] - math.pi, None])

        # 为这一行随后（从左到右）每个角度计算相对于前一个角度的实际偏移量
        # Calculer le biais réel de chaque point suivant par rapport au son point précédent
        for i in range(1, width):
            # 计算偏移量的绝对值
            # Calculer la valeur absolue du biais
            diff = abs(ligne[i][0] - ligne[i - 1][0])
            # 判断偏移是否导致真实值超越arctan的定义域边界
            # Déterminer si ce biais contuit que la valeur réelle de l'angle passe (la multiplication de) le bord de
            # l'ensemble de definition de arctan
            if diff > math.pi / 2:
                # 如果大于Pi/2，则说明偏移导致真实角度突破arctan的定义域边界，则真实的偏移角度应该是Pi - diff
                # Si la valeur > Pi/2, alors le biais constuit le passage, le biais réel doit être Pi - diff
                diff = math.pi - diff
                # 只有两种突破边界的情况：
                # il n'existe que deux cas du passer le bord:
                if ligne[i][0] >= ligne[i - 1][0]:
                    # 上一个角度大于但接近0，下一个角度（反方向）顺时针偏移后为负角度，取模后变成接近pi的角度
                    # l'angle précédent est proche à 0 (mod Pi), l'angle suivant est dans le sens trigonométrique,
                    # et devient proche à Pi après modulo.
                    angle_ligne_R.append([angle_ligne_R[i - 1][0] - diff, None])
                else:
                    # 上一个角度接近pi，下一个角度（正方向）逆时针偏移后超过pi，取模后变成接近0的角度
                    # l'angle précédent est et proche à Pi (mod Pi), l'angle suivant est dans le sens des aiguilles 
                    # d'une montre, et devient proche à 0 après modulo.
                    angle_ligne_R.append([angle_ligne_R[i - 1][0] + diff, None])
            else:
                if ligne[i][0] >= ligne[i - 1][0]:
                    # 下一个角度比上一个角度大，（正方向）逆时针偏移
                    # l'angle suivant est plus grand que le précédent, le biais dans le sens trigonométrique
                    angle_ligne_R.append([angle_ligne_R[i - 1][0] + diff, None])
                else:
                    # 下一个角度比上一个角度小，（反方向）顺时针偏移
                    # l'angle suivant est plus petit que le précédent, le biais dans le sens des aiguilles d'une montre
                    angle_ligne_R.append([angle_ligne_R[i - 1][0] - diff, None])
        # 将当前行储存
        # Enregistrer cette ligne courante
        tab_angle_R.append(angle_ligne_R)

    """修正各点的角度R  y方向
        Corriger ses angles dont valeurs en R pour chaque point (la direction x)"""
    # 获取行的个数
    # Obtenir le nombre de lignes
    length = len(tab_angle_pi)
    # 按列遍历未修正的角度集合，来修正theta_y的数据
    # Parcourir l'ensemble des angles non-corrigé ligne par ligne, pour corriger les données de theta_x
    for j in range(width):
        # 从上到下
        # 初始化每行最开头的角度，奠定随后各点真实角度的偏移基准
        # De haut à bas
        # Initialiser le premier angle d'une ligne. C'est la base de biais de l'angle de point suivant.
        if tab_angle_pi[0][j][1] < math.pi / 2:
            # 如果每行最初的角度是正方向的（即逆时针）（因为在生成tab_angle_pi时每个角度都被模pi）
            # 注意tab_angle_pi[i][j]中每个元素其实是一个含两个元素的np.array，第二个为theta_y
            # Si le premier angle est dans le sens trigonométrique (sens inverse (ou contraire) des aiguilles d'une
            # montre) (car l'angle 'tab_angle_pi' a été modulo Pi après qu'il était généré)
            # Attention, chaque élément dans 'ligne' est en type de 'np.array' qui contient deux éléments, le deuxième 
            # est theta_y
            tab_angle_R[0][j][1] = tab_angle_pi[0][j][1]
        else:
            # 反之为逆方向（顺时针）
            # Sinon, c'est le sens des aiguilles d'une montre
            tab_angle_R[0][j][1] = tab_angle_pi[0][j][1] - math.pi

        # 为这一列随后（从上到下）每个角度计算相对于前一个角度的实际偏移量
        # Calculer le biais réel de chaque point suivant par rapport au son point précédent
        for i in range(1, length):
            # 计算偏移量的绝对值
            # Calculer la valeur absolue du biais
            diff = abs(tab_angle_pi[i][j][1] - tab_angle_pi[i - 1][j][1])
            # 判断偏移是否导致真实值超越arctan的定义域边界
            # Déterminer si ce biais contuit que la valeur réelle de l'angle passe (la multiplication de) le bord de
            # l'ensemble de definition de arctan
            if diff > math.pi / 2:
                # 如果大于pi/2，则说明偏移导致真实角度突破arctan的定义域边界，则真实的偏移角度应该是Pi - diff
                # Si la valeur > Pi/2, alors le biais constuit le passage, le biais réel doit être Pi - diff
                diff = math.pi - diff
                # 只有两种突破边界的情况：
                # il n'existe que deux cas du passer le bord:
                if tab_angle_pi[i][j][1] >= tab_angle_pi[i - 1][j][1]:
                    # 上一个角度大于但接近0，下一个角度（反方向）顺时针偏移后为负角度，取模后变成接近pi的角度
                    # l'angle précédent est proche à 0 (mod Pi), l'angle suivant est dans le sens trigonométrique,
                    # et devient proche à Pi après modulo.
                    tab_angle_R[i][j][1] = tab_angle_R[i - 1][j][1] - diff
                else:
                    # 上一个角度接近pi，下一个角度（正方向）逆时针偏移后超过pi，取模后变成接近0的角度
                    # l'angle précédent est et proche à Pi (mod Pi), l'angle suivant est dans le sens des aiguilles 
                    # d'une montre, et devient proche à 0 après modulo.
                    tab_angle_R[i][j][1] = tab_angle_R[i - 1][j][1] + diff
            else:
                if tab_angle_pi[i][j][1] >= tab_angle_pi[i - 1][j][1]:
                    # 下一个角度比上一个角度，（正方向）逆时针偏移
                    # l'angle suivant est plus grand que le précédent, le biais dans le sens trigonométrique
                    tab_angle_R[i][j][1] = tab_angle_R[i - 1][j][1] + diff
                else:
                    # 下一个角度比上一个角度，（反方向）顺时针偏移
                    # l'angle suivant est plus petit que le précédent, le biais dans le sens des aiguilles d'une montre
                    tab_angle_R[i][j][1] = tab_angle_R[i - 1][j][1] - diff
            """
            if tab_angle_R[i][j][1] is None:
                print("Error, case ({},{})".format(i,j))
            else:
                print(tab_angle_R[i][j][1])
            """
    return tab_angle_R


def afficher_h(_f, _taille=50, _t0=-1, _t1=1):
    _pas = (_t1 - _t0) / _taille
    I = np.arange(_t0, _t1, _pas)
    # diff = plt.figure()
    for y_ in np.arange(_t0, _t1, _pas):  # y fixé
        pos_init = [(x_, y_) for x_ in I]  # ligne horizontale d'ordonnée y dans le carré unité
        pos_finale = [_f(p[0], p[1]) for p in pos_init]

        X = []  # abscisses à plotter
        Y = []  # ordonnées à plotter

        for i in range(len(pos_finale)):
            X.append(pos_finale[i][0])
            Y.append(pos_finale[i][1])

        plt.plot(X, Y)


def afficher_h_vecteur(_f_sym):
    I = np.arange(-1, 1, 0.01)
    df_sym_temps = diff_sym(_f_sym)

    def diff(_point):
        return diff_num(df_sym_temps, _point)

    for y_ in np.arange(-1, 1, 0.01):  # y fixé
        pos_init = [(x_, y_) for x_ in I]  # ligne horizontale d'ordonnée y dans le carré unité

        # pos_init = [(x_,y_) for x_ in I for y_ in I]
        pos_finale = [diff(p) for p in pos_init]

        X = []  # abscisses à plotter
        Y = []  # ordonnées à plotter

        for i in range(len(pos_finale)):
            X.append(pos_finale[i][0][0])
            Y.append(pos_finale[i][1][0])

        """
        x1min = -1
        x1max = 1
        dx1 = 0.01
        x1 = np.arange(x1min, x1max, dx1)
        x2min = -1.
        x2max = 1
        dx2 = 0.01
        x2 = np.arange(x2min, x2max, dx2)
        """
        plt.quiver(I, I, X, Y)
        # Calculer le champ de vecteur
    # XX1, XX2 = np.meshgrid(I, I)


def afficher_v(_f, _taille=50, _t0=-1, _t1=1):
    _pas = (_t1 - _t0) / _taille
    I = np.arange(_t0, _t1, _pas)
    # diff = plt.figure()
    for x_ in np.arange(_t0, _t1, _pas):  # x fixé
        pos_init = [(x_, y_) for y_ in I]
        pos_finale = [_f(p[0], p[1]) for p in pos_init]  # juste une translation de la même courbe
        # pos_finale = [f(p[0],p[1], ALPHA, BETA) for p in pos_init] #encore plus bizarre

        X = []
        Y = []

        for i in range(len(pos_finale)):
            X.append(pos_finale[i][0])
            Y.append(pos_finale[i][1])

        plt.plot(X, Y)


def test(_switch_f=0, _a=0.2, _b=5, _point_test=(0.99, -0.99), _theta=math.pi * 5, _t0=-1, _t1=1, _taille=50):
    _f = None
    if _switch_f == 0:
        print("###### Test du f_ex({},{}) dans [{},{}], avec la fin {} ######".format(_a, _b, _t0, _t1, _taille))
        _f = f_ex(_a, _b)
        _f_sym, _f_num = _f
        _df_sym = diff_sym(_f_sym)
        _df_num = diff_num(_df_sym, _point_test)
        print("f_sym = {}\ndf_sym = {}\ndf_num en point {} = {}\n".format(_f_sym, _df_sym, _point_test, _df_num))

        afficher_h(_f_num, _taille)
        afficher_v(_f_num, _taille)
        plt.show()

        pas = (_t1 - _t0) / _taille
        I2 = [[(-1 + i * pas, 1 - j * pas) for i in range(_taille)] for j in range(_taille)]
        tab_angle = tab_angle_num(I2, _df_sym)
        print("tab_angle = {}".format(tab_angle))
        print("###### Fin du test ######")

        return [_f_sym, _f_num, _df_sym, tab_angle]

    elif _switch_f == 1:
        print(
            "###### Test des f_ex2({},{}) et r_ex(f_ex2,{}) dans [{},{}], avec la fin {} ######".format(_a, _b, _theta,
                                                                                                        _t0, _t1,
                                                                                                        _taille))
        _f = f_ex2(_a, _b)
        _rf_sym, _rf_num = r_ex(_theta, _f[0], _f[1])
        _drf_sym = diff_sym(_rf_sym)
        _drf_num = diff_num(_drf_sym, _point_test)
        print("f_sym = {}\ndrf_sym = {}\ndrf_num en point {} = {}\n".format(_rf_sym, _drf_sym, _point_test, _drf_num))

        afficher_h(_rf_num, _taille)
        afficher_v(_rf_num, _taille)
        plt.show()

        pas = (_t1 - _t0) / _taille
        I2 = [[(-1 + i * pas, 1 - j * pas) for i in range(_taille)] for j in range(_taille)]
        tab_angle = tab_angle_num(I2, _drf_sym)
        print("tab_angle = {}".format(tab_angle))
        print("tab_angle[{}] = {}".format(_taille // 2, tab_angle[_taille // 2]))
        print("###### Fin du test ######")

        return [_rf_sym, _rf_num, _drf_sym, tab_angle]


test(0)
print("\n\n")
test(1, _theta=20 * math.pi)
