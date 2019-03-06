import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import scipy
from scipy.misc import derivative

# from sklearn.preprocessing import normalize #对matrix取范数

# variables
x, y = sp.symbols("x y")


def f_ex(a, b):
    """
    返回同一个函数的两个形式，第一个用sympy符号表达，第二个用python函数表达
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

    return f_num


def r_ex(_theta, _f_num):
    # mat = np.array([[math.cos(_theta), -math.sin(_theta)], [math.cos(_theta), math.cos(_theta)]])

    def f_new(_x, _y):
        # temps = np.array(_f_num(_x, _y))
        temps = _f_num(_x, _y)
        # return np.dot(mat, temps)
        return math.cos(temps) * _x - math.sin(temps) * _y, math.sin(temps) * _x + math.cos(temps) * _y

    return f_new


def diff_sym(_f):
    """
    对一个符号表达函数f：I^2->I^2求其微分表达式。注意返回的矩阵是一个一维列表，先从上到下，再从左到右
    :param _f:
    :return: | &φ1/&x  &φ1/&y |
              | &φ2/&x  &φ2/&y |
    """
    return sp.Matrix([[sp.diff(_f[0], x), sp.diff(_f[0], y)], [sp.diff(_f[1], x), sp.diff(_f[1], y)]])


def normer(_diff_num):
    row_sums = _diff_num.sum(axis=1)
    normed_mat = _diff_num / row_sums[:, np.newaxis]
    return normed_mat


def diff_num(_f_sym, _point):
    """
    对给出的一个2*2的微分表达式，和某一个给定的点（a，b）进行数值替换（x<-a, y<-b），并进行取模
    :param _f_sym:
    :param _point:
    :return: 返回的表格是以“行”为单位的，二维表格
    """
    x1 = _f_sym[0].subs({x: _point[0], y: _point[1]})
    x2 = _f_sym[1].subs({x: _point[0], y: _point[1]})
    y1 = _f_sym[2].subs({x: _point[0], y: _point[1]})
    y2 = _f_sym[3].subs({x: _point[0], y: _point[1]})

    mat = np.array([[x1, x2], [y1, y2]])
    return normer(mat)


def angle_num(_df_num):
    """
    接收一个2*2的矩阵，返回其每列对应的角度
    :param _df_num:
    :return:
    """
    a1 = 0
    a2 = 0
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
    return theta_h, theta_v


def diff_via_angle_num(_angle_num):
    """
    接受两个角度，返回对应的微分矩阵（以行为单位）
    :param _angle_num:
    :return:
    """

    a1 = math.sin(_angle_num[0])
    a3 = math.sin(_angle_num[1])
    a2 = math.cos(_angle_num[0])
    a4 = math.cos(_angle_num[1])
    return [[a1, a2], [a3, a4]]


def afficher_h(_f):
    I = np.arange(-1, 1, 0.01)
    diff = plt.figure()
    for y_ in np.arange(-1, 1, 0.01):  # y fixé
        pos_init = [(x_, y_) for x_ in I]  # ligne horizontale d'ordonnée y dans le carré unité

        pos_finale = [_f(p[0], p[1]) for p in pos_init]

        X = []  # abscisses à plotter
        Y = []  # ordonnées à plotter

        for i in range(len(pos_finale)):
            X.append(pos_finale[i][0])
            Y.append(pos_finale[i][1])

        plt.plot(X, Y)


def afficher_v(_f):
    I = np.arange(-1, 1, 0.01)
    for x_ in np.arange(-1, 1, 0.01):  # x fixé
        pos_init = [(x_, y_) for y_ in I]

        pos_finale = [_f(p[0], p[1]) for p in pos_init]  # juste une translation de la même courbe
        # pos_finale = [f(p[0],p[1], ALPHA, BETA) for p in pos_init] #encore plus bizarre

        X = []
        Y = []

        for i in range(len(pos_finale)):
            X.append(pos_finale[i][0])
            Y.append(pos_finale[i][1])

        plt.plot(X, Y)


f, fnum = f_ex(0.2, 5)
print("f= ", f, "\n---")
df = diff_sym(f)
print("df= ", df, "\n---")
point = [0.5, 0.5]
df_num = diff_num(df, point)
print("df_num, en poinr (0.5,0.5)= ", df_num, "\n---")
angles = angle_num(df_num)
print("angles= ", angles, "\n---")
df_angles = diff_via_angle_num(angles)
print("diff via angles= ", df_angles)
afficher_h(fnum)
plt.show()

"""
afficher_h(fnum)
plt.show()
afficher_v(fnum)
plt.show()
ro_num = r_ex(math.pi / 4, fnum)
print(ro_num(2, 3))
afficher_h(ro_num)
plt.show()
"""
"""
fnum2 = f_ex2(0.2, 5)
ro_num2 = r_ex(0,fnum2)
print(ro_num2(2, 3))
afficher_h(ro_num2)
plt.show()
afficher_v(ro_num2)
plt.show()
"""
