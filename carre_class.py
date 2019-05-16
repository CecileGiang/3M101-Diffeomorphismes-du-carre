import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import sympy as sp


class DiffeoInfini:
    """
    Classe de fonction en C-diff-infini R^2->R^2
    Ici, on pense que un diffeomorphisme est un objet dans le classe de fonction en C-diff-infini R^2->R^2.
    On declare un tel objet par une expression de sympy, qui prend deux arguments x et y, et retourne deux valeurs.
    Par ex.:
        x,y=sp.symbols("x y")
        expr=(x+y,x*y)
        ex=fonc_diff_infini(expr,(x,y))
    """

    def __init__(self, expr, expr_reci=None, t0=-1., t1=1., snb=None, vars_sym=(sp.symbols("x y"))):
        """
        Pour creer une instance d'un diffeomorphisme de I^2 a I^2, il faut donner son expression mathematique,
        c'est-a-dire, une expression symbolique, ou on represente ses deux variables par x et y par default, et il faut
        aussi preciser le domaine I^2 sous forme [t0, t1]^2. De plus, pour accederer le calcul de sa fonction reciproque
        on peut se donner son expression symbolique comme une option.
        :param expr: l'expression symbolique du diffeomorphisme
        :param expr_reci: l'expression symbolique du diffeomorphisme reciproque
        :param t0: valeur minimale dans l'interval I
        :param t1: valeur maximale dans l'interval I
        :param snb: nombre de l'échantillonnage par default sur une dimention. Ex. t0=-1, t1=1, snb=5, alors I = [-1, 1]
                    sera echantillonne par [-1, -0.5, 0, -0.5, 1]
        :param vars_sym: les symboles qui representent les deux variables du diffeomorphisme
        """
        """
        ((*) signifier cette variable peut etre None, (#)signifier cette variable devient None si l'interval est change)
        Variables de base :
        __expr: l'expression symbolique de ce diffeomorphisme
        __expr_reci: (*)l'expression symbolique du diffeomorphisme reciproque
        __t0, __t1: (#)etant respectivement -1 et 1 par default, consistent a l'interval I tel que I = [__t0, __t1]
        __x, __y: etant respectivement x et y par default, representent les deux variables du diffeomophisme
        __num: la fonction de python correspondant a expr
        __num_reci: (*)la fonction de python correspondant a expr_reci
        """
        self.__expr = expr
        self.__expr_reci = expr_reci
        self.__t0, self.__t1 = t0, t1
        self.__x, self.__y = vars_sym
        self.__num = sp.lambdify((self.__x, self.__y), self.__expr, "numpy")
        self.__num_reci = sp.lambdify((self.__x, self.__y), self.__expr_reci,
                                      "numpy") if self.__expr_reci is not None else None
        """
        Variables sur le differentiel :
        __df_sym: l'expression symbolique du differentiel de ce diffeomorphisme
        __df_num: la fonction de python correspondant a df_sym
        __df_reci_sym: (*)l'expression symbolique du differentiel du diffeomorphisme reciproque
        __df_reci_num: (*)la fonction de python correspondant a df_reci_sym
        """
        self.__df_sym = self.df_dim2_sym(self.__expr, self.__x, self.__y)
        self.__df_num = sp.lambdify((self.__x, self.__y), self.__df_sym, "numpy")
        self.__df_reci_sym = self.df_dim2_sym(self.__expr_reci, self.__x,
                                              self.__y) if self.__expr_reci is not None else None
        self.__df_reci_num = sp.lambdify((self.__x, self.__y), self.__df_reci_sym,
                                         "numpy") if self.__df_reci_sym is not None else None
        """
        Variables sur les resultats sous forme de tableau :
        snb: nombre de l'échantillonnage par default sur une dimention
        _plan: (#)meshes grid d'un plan, sous forme de meshgrid en 3 dimentions: (x ou y, ligne, colonne)
        _tab_f: (#)f(_plan), sous meme forme que _plan
        _tab_df: (#)df(_plan), sous forme de meshgrid en 4 dimenttions: (∂f1 ou ∂f2, ∂x ou ∂y, ligne, colonne)
        _tab_points_reci: (#)f(_tab_points_reci) ≈ _plan, sous meme forme que plan
        _tab_f_points_reci: (#)f(_tab_points_reci), sous meme forme que plan
        _tab_df_points_reci: (#)df(_tab_points_reci), sous meme forme que plan
        _tab_angles_R: (#)les angles calcules a partir de _tab_df_points_reci, sous forme de meshgrid en 3 dimentions:
                        (horizontal ou vertical, ligne, colonne)
        """
        self.snb = snb or int((t1 - t0) * 25)
        self._plan = None
        self._tab_f = None
        self._tab_df = None
        self._tab_points_reci = None
        self._tab_f_points_reci = None
        self._tab_df_points_reci = None
        self._tab_angles_R = None

    def change_domain(self, t0=None, t1=None):
        """
        Changer le domaine du diffeomorphisme, et changer les donnees qui le consernent
        :param t0: float
        :param t1: float
        :return: boolean : True si le domaine est change, False sinon
        """
        flag = False
        if t0 is not None and t0 != self.__t0:
            self.__t0 = t0
            flag = True
        if t1 is not None and t1 != self.__t1:
            self.__t1 = t1
            flag = True
        if flag:
            self._plan = None
            self._tab_points_reci = None
            self._tab_f = None
            self._tab_df = None
            self._tab_angles_R = None
        return flag

    def plan(self, snb=None):
        """
        Retourner deux tableaux qui sont les 'rastérisations' (feuillages) d'un plan traitees par numpy.meshgrid.
        Attention, la structure de ces deux tableaux sont specifiques. Veuilliez-vous afficher ces deux tableaux pour
        la connaitre. C'est pour faciliter le calcul d'apres.
        :param snb: int
        :return: [x ou y, ligne, colonne]
        """
        taille = snb or self.snb
        if self._plan is None or taille != len(self._plan):
            axe = np.linspace(self.__t0, self.__t1, taille)
            self._plan = np.meshgrid(axe, axe)
        return self._plan

    @staticmethod
    def df_dim2_sym(expr, sym_x, sym_y):
        """
        对一个符号表达函数f：I^2->I^2求其微分表达式。注意返回的矩阵是一个一维列表，先从上到下，再从左到右
        Pour une fonction symbolique f：I^2->I^2, on calcul son différentiel.
        Attention, le résultat est un liste en 1 dimensionla, en lisant la matrice de différentiel de haut à bas,
        et de gauche à droit
        :param expr:
        :param sym_x:
        :param sym_y:
        :return: | ∂f1/∂x  ∂f1/∂y |
                 | ∂f2/∂x  ∂f2/∂y |
        """
        df_sym = sp.Matrix([[sp.diff(expr[0], sym_x), sp.diff(expr[0], sym_y)],
                            [sp.diff(expr[1], sym_x), sp.diff(expr[1], sym_y)]])
        return df_sym

    def tab_f(self, snb=None):
        """
        En prennant un plan idendite, calculuer l'image de son chaque point dans l'ensemble arrive
        :param snb: int
        :return: [x ou y, ligne, colonne]
        """
        taille = snb or self.snb
        if self._tab_f is None or taille != len(self._tab_f[0]):
            axe_x, axe_y = self.plan(taille)
            tab_x, tab_y = self.__num(axe_x, axe_y)
            self._tab_f = (tab_x, tab_y)
        return self._tab_f

    def tab_df(self, snb=None):
        """
        En utilisant un plan feuillage par numpy.meshgrid, on symplifie le code pour calculer un tableau de matrice
        jacobienne de chaque point dans le plan.
        Attention, le resultat retourne est en la structure de meshgrid. Voir detailles dans la partie ":return"
        :param snb: int
        :return: le resultat est en numpy.array. Soit [[a,b],[c,d]] la matrice jacobienne du diffeomorphisme dans
        un point, resultat[0][0] contient les a de chaque point ET DANS LA TRUCTURE DE numpy.meshgrid. De meme,
        resultat[0][1] contient les c, resultat[1][0] contient les b, et resultat[1][1] contient les d.
        """
        taille = snb or self.snb
        if self._tab_df is None or taille != len(self._tab_df[0][0]):
            axe_x, axe_y = self.plan(taille)
            self._tab_df = self.__df_num(axe_x, axe_y)
        return self._tab_df

    def f_reci(self, x_num, y_num):
        """
        Calculer l'inverse de (x_num, y_num) par le diffeomorphisme
        :param x_num: float
        :param y_num: float
        :return: float: f^-1(x_num, y_num)
        """
        if self.__num_reci is not None:
            return self.__num_reci(x_num, y_num)
        else:
            return None

    def load_points_reci(self, path, t0, t1, struc="tab"):
        if t0 != self.__t0 or t1 != self.__t1:
            print("Error: t0 ou t1 ne correspond pas au domain de ce diffeomorphime")
        else:
            if struc == "grid":
                self._tab_points_reci = np.load(path)
            else:
                temp = np.load(path)
                tab_x, tab_y = [], []
                for ligne in temp:
                    ligne_x, ligne_y = [], []
                    for point in ligne:
                        ligne_x.append(point[0])
                        ligne_y.append(point[1])
                    tab_x.append(ligne_x)
                    tab_y.append(ligne_y)
                self._tab_points_reci = np.array([tab_x, tab_y])

    def tab_points_reci(self, snb=None, multi=10):
        taille = snb or self.snb
        if self._tab_points_reci is None:
            if self.__num_reci is None:
                pass
            axe_x, axe_y = self.plan(taille)
            tab_x, tab_y = self.__num_reci(axe_x, axe_y)
            self._tab_points_reci = (tab_x, tab_y)

        return self._tab_points_reci

    def tab_f_points_reci(self, snb=None, multi=10):
        taille = snb or self.snb
        if self._tab_f_points_reci is None or taille != len(self._tab_f_points_reci[0]):
            tab_reci_x, tab_reci_y = self.tab_points_reci(taille, multi)
            self._tab_f_points_reci = self.__num(tab_reci_x, tab_reci_y)
        return self._tab_f_points_reci

    def tab_df_points_reci(self, snb=None, multi=10):
        taille = snb or self.snb
        if self._tab_df_points_reci is None or taille != len(self._tab_df_points_reci[0][0]):
            tab_reci_x, tab_reci_y = self.tab_points_reci(taille, multi)
            self._tab_df_points_reci = self.__df_num(tab_reci_x, tab_reci_y)
        return self._tab_df_points_reci

    def tab_angles_R(self, snb=None, multi=10):
        """
        Calculer les angles des vecteurs dans le champ de vecteur du diffeomorphisme des directions horizontale et
        verticale. Pour chaque point dans l'ensemble arrive, son vecteur horizontal est [∂f1/∂x, ∂f2/∂x], et son angle
        horizontal est (∂f2/∂x / ∂f1/∂x). De meme, son vectuer vertical est [∂f1/∂y, ∂f2/∂y], et son angle vertical
        est (∂f2/∂y / ∂f1/∂y)
        :param snb: int
        :param multi:
        :return: [horizontal ou vertical, ligne, colonne]
        """
        taille = snb or self.snb

        def modulo_2pi(x_):
            """
            En prenant une valeur reelle x_, retourner y_ tel que y dans [-pi, pi] et x_ mod 2pi = y_
            :param x_: float
            :return: float
            """
            y_ = x_
            while y_ > math.pi:
                y_ -= 2 * math.pi
            while y_ < -math.pi:
                y_ += 2 * math.pi
            return y_

        def corrigeur(tab):
            tab_R = []
            for ligne in tab:
                ligne_R = [ligne[0]]
                for angle in ligne[1:]:
                    prec = ligne_R[-1]
                    ligne_R.append(prec + modulo_2pi(angle - prec))
                tab_R.append(ligne_R)
            return np.array(tab_R)

        if self._tab_angles_R is None or taille != len(self._tab_angles_R[0]):
            tab_df = self.tab_df_points_reci(taille, multi)
            tab_angles_x_2pi = np.arctan2(tab_df[1][0], tab_df[0][0])
            tab_angles_y_2pi = np.arctan2(tab_df[1][1], tab_df[0][1])

            tab_angles_x_R = corrigeur(tab_angles_x_2pi)
            tab_angles_y_R = corrigeur(tab_angles_y_2pi.T) - math.pi / 2

            self._tab_angles_R = np.array([tab_angles_x_R, tab_angles_y_R])
        return self._tab_angles_R

    def _distance(self, x_, y_, tab_x_mesh, tab_y_mesh):
        """
        Pour chaque point dans l'ensemble donne (tab_x_mesh, tab_y_mesh), calculer la distance euclidienne entre lui et
        le point (x_,y_)
        :param x_:
        :param y_:
        :param tab_x_mesh:
        :param tab_y_mesh:
        :return:
        """
        tempx = tab_x_mesh - x_
        tempy = tab_y_mesh - y_
        tab_2d = []
        for i in range(len(tab_x_mesh)):
            tab_2d.append(np.sqrt(tempx[i] ** 2 + tempy[i] ** 2))
        return np.array(tab_2d)

    def _classifier_tab(self, tab_dis, tab_x_mesh, tab_y_mesh, pas, n):
        """
        Classier les points dans l'ensemble donne (tab_x_mesh, tab_y_mesh), par leur distance:
        [0, pas/2) [pas/2, pas + pas/2) [pas + pas/2, 2*pas + pas/2) ... [(n-2)*pas + pas/2, (n-1)*pas + pas/2)
        (>=(n-1)*pas + pas/2)
        :param tab_dis: les distances des points dans (tab_x_mesh, tab_y_mesh):
         tab_2d[i,j]->tab_x_mesh[i,j],tab_y_mesh[i,j]
        :param tab_x_mesh:
        :param tab_y_mesh:
        :param pas: la distance entre deux points adjacents dans une meme ligne de la grille du plan [t0, t1]^2
        :param n: le nombre des niveaux de classification
        :return:
        """
        tab_res_x = [[] for i in range(n + 1)]
        tab_res_y = [[] for i in range(n + 1)]
        tab_dis_bis = (tab_dis - pas / 2) // pas + 1
        for i in range(len(tab_dis_bis)):
            for j in range(len(tab_dis_bis[i])):
                niveau = int(tab_dis_bis[i, j])
                if niveau >= n:
                    tab_res_x[n].append(tab_x_mesh[i, j])
                    tab_res_y[n].append(tab_y_mesh[i, j])
                else:
                    tab_res_x[niveau].append(tab_x_mesh[i, j])
                    tab_res_y[niveau].append(tab_y_mesh[i, j])
        return tab_res_x, tab_res_y

    def _classifier_points_cles(self, tab_dis, tab_x_mesh, tab_y_mesh, pas, n):
        tab_res_x = [[] for i in range(n + 1)]
        tab_res_y = [[] for i in range(n + 1)]
        tab_dis_bis = tab_dis // pas + 1
        for i in range(len(tab_dis_bis)):
            for j in range(len(tab_dis_bis[i])):
                niveau = int(tab_dis_bis[i, j])
                if tab_dis[i, j] >= pas / 2 + niveau * pas:
                    niveau += 1
                elif tab_dis[i, j] <= pas / 2 + (niveau - 1) * pas:
                    niveau -= 1
                if niveau >= n:
                    tab_res_x[n].append(tab_x_mesh[i, j])
                    tab_res_y[n].append(tab_y_mesh[i, j])
                else:
                    tab_res_x[niveau].append(tab_x_mesh[i, j])
                    tab_res_y[niveau].append(tab_y_mesh[i, j])
        return tab_res_x, tab_res_y

    def tab_inverse(self, t0=-1, t1=1, snb=None, multi=10):
        taille = snb or self.snb
        axe = np.linspace(t0, t1, taille)
        axe_x, axe_y = np.meshgrid(axe, axe)
        axe2 = np.linspace(t0, t1, taille * multi)
        axe2_x, axe2_y = np.meshgrid(axe2, axe2)
        f = self.__num
        ens_arrive_x, ens_arrive_y = f(axe2_x, axe2_y)

        pas = (t1 - t0) / (taille - 1)
        n = math.ceil((taille // 2) * math.sqrt(2))

        def ajustement(x_, y_, tab_x_mesh, tab_y_mesh):
            dis = np.inf
            test = (pas ** 2) / 4
            val_x, val_y = None, None
            tab_x_new, tab_y_new = [], []
            for i in range(len(tab_x_mesh)):
                tempx, tempy = [], []
                for j in range(len(tab_x_mesh[i])):
                    x_ori, y_ori = tab_x_mesh[i, j], tab_y_mesh[i, j]
                    fx, fy = f(x_ori, y_ori)
                    d = (fx - x_) ** 2 + (fy - y_) ** 2
                    if d < dis:
                        dis = d
                        val_x = x_ori
                        val_y = y_ori
                    if d >= test:
                        tempx.append(x_ori)
                        tempy.append(y_ori)
                tab_x_new.append(tempx)
                tab_y_new.append(tempy)
            return val_x, val_y, tab_x_new, tab_y_new

        tab_dis_cles = ex._distance(-1, -1, axe_x, axe_y)
        class_x, class_y = ex._classifier_points_cles(tab_dis_cles, axe_x, axe_y, pas, n)
        tab_dis = ex._distance(-1, -1, ens_arrive_x, ens_arrive_y)
        tab_inv_x, tab_inv_y = ex._classifier_tab(tab_dis, axe2_x, axe2_y, pas, n)
        for niveau in range(len(class_x)):
            for i in range(len(class_x[niveau])):
                pass
        pass

        """
        ens_depart = [(x_, y_) for x_ in np.linspace(t0, t1, taille2) for y_ in np.linspace(t0, t1, taille2)]
        ens_arrive = [self._num(p[0], p[1]) for p in ens_depart]
        ens_critere = [[(x_, y_) for x_ in np.linspace(t0, t1, taille)] for y_ in np.linspace(t0, t1, taille)]
        ens_inverse = []
        for ligne in ens_critere:
            ligne_inverse = []
            for pc in ligne:
                dif_min = np.inf
                point_proche = None
                for i in range(taille2 ** 2):
                    dif = (ens_arrive[i][0] - pc[0]) ** 2 + (ens_arrive[i][1] - pc[1]) ** 2
                    if dif < dif_min:
                        dif_min = dif
                        point_proche = ens_depart[i]
                ligne_inverse.append(point_proche)
            ens_inverse.append(ligne_inverse)
        return ens_inverse
        """

    def trace(self, temps=1, snb=None, multi=10, precision=0.005, methode="rk", symetrique=False):
        """
        A partir les angles en moment temps, tracer l'image du diffeomorphisme dont snb courbes horizontals, snb
        courbes verticals. L'ensemble des angles est de taille snb*multi. precision est le pas de trace. methode en
        choix signifie la methode mathematique utilisee pour tracer l'image. symetrique indique si on beneficie la
        symetrie du diffeomorphisme
        :param temps: float dans [0, 1]
        :param snb: int
        :param multi:
        :param precision:
        :param methode:
        :param symetrique:
        :return:
        """
        taille = snb or self.snb
        t0, t1 = self.__t0, self.__t1
        tab_angles = self.tab_angles_R(taille, multi) * temps
        axe = np.linspace(t0, t1, taille)
        res = []

        def find_sim_points(x_, y_):
            pas = (t1 - t0) / (taille - 1)
            kx0 = int((x_ - t0) // pas)
            if kx0 >= taille:
                kx0 = taille - 1
            kx1 = kx0 + 1 if kx0 < taille - 1 else kx0
            t = (x_ - t0 - kx0 * pas) / pas
            ky0 = int((y_ - t0) // pas)
            if ky0 >= taille:
                ky0 = taille - 1
            ky1 = ky0 + 1 if ky0 < taille - 1 else ky0
            s = (y_ - t0 - ky0 * pas) / pas
            return (kx0, ky0), (kx1, ky0), (kx0, ky1), (kx1, ky1), t, s

        def angle_moyen(direc, p00, p10, p01, p11, t, s):
            if direc == 'h':
                a00 = np.array([tab_angles[0, p00[1], p00[0]]])
                a10 = np.array([tab_angles[0, p10[1], p10[0]]])
                a01 = np.array([tab_angles[0, p01[1], p01[0]]])
                a11 = np.array([tab_angles[0, p11[1], p11[0]]])
            else:
                a00 = np.array([tab_angles[1, p00[0], p00[1]]])
                a10 = np.array([tab_angles[1, p10[0], p10[1]]])
                a01 = np.array([tab_angles[1, p01[0], p01[1]]])
                a11 = np.array([tab_angles[1, p11[0], p11[1]]])
            angle = (1 - t) * ((1 - s) * a00 + s * a01) + t * ((1 - s) * a10 + s * a11)
            return angle

        def angle(direc, x_, y_):
            p00, p10, p01, p11, t, s = find_sim_points(x_, y_)
            return angle_moyen(direc, p00, p10, p01, p11, t, s)

        def runge_kutta_demi(direc, demi, sens):
            tab = []
            if direc == 'h':
                if sens == "direct":
                    for y_ in demi:
                        tab_trace_x, tab_trace_y = [-1.0], [y_]
                        while tab_trace_x[-1] < t1:
                            p_n1 = [tab_trace_x[-1], tab_trace_y[-1]]  # 初始点
                            a1 = angle(direc, p_n1[0], p_n1[1])  # 初始点的斜率
                            p_n2 = [p_n1[0] + precision / 2 * math.cos(a1),
                                    p_n1[1] + precision / 2 * math.sin(a1)]  # 用a1计算的中点
                            a2 = angle(direc, p_n2[0], p_n2[1])  # 用a1计算的中点的斜率
                            p_n3 = [p_n1[0] + precision / 2 * math.cos(a2),
                                    p_n1[1] + precision / 2 * math.sin(a2)]  # 用a2计算的中点
                            a3 = angle(direc, p_n3[0], p_n3[1])  # 用a2计算的中点的斜率
                            p_n4 = [p_n1[0] + precision * math.cos(a3),
                                    p_n1[1] + precision * math.sin(a3)]  # 用a3计算的终点
                            a4 = angle(direc, p_n4[0], p_n4[1])  # 用a3计算的终点斜率
                            a_m = (a1 + 2 * a2 + 2 * a3 + a4) / 6  # 加权平均斜率
                            tab_trace_x.append(p_n1[0] + precision * math.cos(a_m))
                            tab_trace_y.append(p_n1[1] + precision * math.sin(a_m))
                        tab_trace_x.pop(-1)
                        tab_trace_y.pop(-1)
                        tab.append([tab_trace_x, tab_trace_y])
                else:
                    for y_ in demi:
                        tab_trace_x, tab_trace_y = [1.0], [y_]
                        while tab_trace_x[-1] >= t1:
                            p_n1 = [tab_trace_x[-1], tab_trace_y[-1]]  # 初始点
                            a1 = angle(direc, p_n1[0], p_n1[1])  # 初始点的斜率
                            p_n2 = [p_n1[0] - precision / 2 * math.cos(a1),
                                    p_n1[1] - precision / 2 * math.sin(a1)]  # 用a1计算的中点
                            a2 = angle(direc, p_n2[0], p_n2[1])  # 用a1计算的中点的斜率
                            p_n3 = [p_n1[0] - precision / 2 * math.cos(a2),
                                    p_n1[1] - precision / 2 * math.sin(a2)]  # 用a2计算的中点
                            a3 = angle(direc, p_n3[0], p_n3[1])  # 用a2计算的中点的斜率
                            p_n4 = [p_n1[0] - precision * math.cos(a3),
                                    p_n1[1] - precision * math.sin(a3)]  # 用a3计算的终点
                            a4 = angle(direc, p_n4[0], p_n4[1])  # 用a3计算的终点斜率
                            a_m = (a1 + 2 * a2 + 2 * a3 + a4) / 6  # 加权平均斜率
                            tab_trace_x.append(p_n1[0] - precision * math.cos(a_m))
                            tab_trace_y.append(p_n1[1] - precision * math.sin(a_m))
                        tab_trace_x.pop(-1)
                        tab_trace_y.pop(-1)
                        tab.append([tab_trace_x, tab_trace_y])
            else:
                if sens == "direct":
                    for x_ in demi:
                        tab_trace_x, tab_trace_y = [x_], [-1.0]
                        while tab_trace_y[-1] < t1:
                            p_n1 = [tab_trace_x[-1], tab_trace_y[-1]]  # 初始点
                            a1 = angle(direc, p_n1[0], p_n1[1])  # 初始点的斜率
                            p_n2 = [p_n1[0] - precision / 2 * math.sin(a1),
                                    p_n1[1] + precision / 2 * math.cos(a1)]  # 用a1计算的中点
                            a2 = angle(direc, p_n2[0], p_n2[1])  # 用a1计算的中点的斜率
                            p_n3 = [p_n1[0] - precision / 2 * math.sin(a2),
                                    p_n1[1] + precision / 2 * math.cos(a2)]  # 用a2计算的中点
                            a3 = angle(direc, p_n3[0], p_n3[1])  # 用a2计算的中点的斜率
                            p_n4 = [p_n1[0] - precision * math.sin(a3),
                                    p_n1[1] + precision * math.cos(a3)]  # 用a3计算的终点
                            a4 = angle(direc, p_n4[0], p_n4[1])  # 用a3计算的终点斜率
                            a_m = (a1 + 2 * a2 + 2 * a3 + a4) / 6  # 加权平均斜率
                            tab_trace_x.append(p_n1[0] - precision * math.sin(a_m))
                            tab_trace_y.append(p_n1[1] + precision * math.cos(a_m))
                        tab.append([tab_trace_x, tab_trace_y])
                else:
                    for x_ in demi:
                        tab_trace_x, tab_trace_y = [x_], [1.0]
                        while tab_trace_y[-1] >= t1:
                            p_n1 = [tab_trace_x[-1], tab_trace_y[-1]]  # 初始点
                            a1 = angle(direc, p_n1[0], p_n1[1])  # 初始点的斜率
                            p_n2 = [p_n1[0] + precision / 2 * math.sin(a1),
                                    p_n1[1] - precision / 2 * math.cos(a1)]  # 用a1计算的中点
                            a2 = angle(direc, p_n2[0], p_n2[1])  # 用a1计算的中点的斜率
                            p_n3 = [p_n1[0] + precision / 2 * math.sin(a2),
                                    p_n1[1] - precision / 2 * math.cos(a2)]  # 用a2计算的中点
                            a3 = angle(direc, p_n3[0], p_n3[1])  # 用a2计算的中点的斜率
                            p_n4 = [p_n1[0] + precision * math.sin(a3),
                                    p_n1[1] - precision * math.cos(a3)]  # 用a3计算的终点
                            a4 = angle(direc, p_n4[0], p_n4[1])  # 用a3计算的终点斜率
                            a_m = (a1 + 2 * a2 + 2 * a3 + a4) / 6  # 加权平均斜率
                            tab_trace_x.append(p_n1[0] + precision * math.sin(a_m))
                            tab_trace_y.append(p_n1[1] - precision * math.cos(a_m))
                        tab.append([tab_trace_x, tab_trace_y])
            return tab

        def runge_kutta(direc):
            tab = []
            if direc == 'h':
                for y_ in axe:
                    tab_trace_x, tab_trace_y = [-1.0], [y_]
                    while tab_trace_x[-1] < t1:
                        p_n1 = [tab_trace_x[-1], tab_trace_y[-1]]  # 初始点
                        a1 = angle(direc, p_n1[0], p_n1[1])  # 初始点的斜率
                        p_n2 = [p_n1[0] + precision / 2 * math.cos(a1),
                                p_n1[1] + precision / 2 * math.sin(a1)]  # 用a1计算的中点
                        a2 = angle(direc, p_n2[0], p_n2[1])  # 用a1计算的中点的斜率
                        p_n3 = [p_n1[0] + precision / 2 * math.cos(a2),
                                p_n1[1] + precision / 2 * math.sin(a2)]  # 用a2计算的中点
                        a3 = angle(direc, p_n3[0], p_n3[1])  # 用a2计算的中点的斜率
                        p_n4 = [p_n1[0] + precision * math.cos(a3),
                                p_n1[1] + precision * math.sin(a3)]  # 用a3计算的终点
                        a4 = angle(direc, p_n4[0], p_n4[1])  # 用a3计算的终点斜率
                        a_m = (a1 + 2 * a2 + 2 * a3 + a4) / 6  # 加权平均斜率
                        tab_trace_x.append(p_n1[0] + precision * math.cos(a_m))
                        tab_trace_y.append(p_n1[1] + precision * math.sin(a_m))
                    tab.append([tab_trace_x, tab_trace_y])
            else:
                for x_ in axe:
                    tab_trace_x, tab_trace_y = [x_], [-1.0]
                    while tab_trace_y[-1] < t1:
                        p_n1 = [tab_trace_x[-1], tab_trace_y[-1]]  # 初始点
                        a1 = angle(direc, p_n1[0], p_n1[1])  # 初始点的斜率
                        p_n2 = [p_n1[0] - precision / 2 * math.sin(a1),
                                p_n1[1] + precision / 2 * math.cos(a1)]  # 用a1计算的中点
                        a2 = angle(direc, p_n2[0], p_n2[1])  # 用a1计算的中点的斜率
                        p_n3 = [p_n1[0] - precision / 2 * math.sin(a2),
                                p_n1[1] + precision / 2 * math.cos(a2)]  # 用a2计算的中点
                        a3 = angle(direc, p_n3[0], p_n3[1])  # 用a2计算的中点的斜率
                        p_n4 = [p_n1[0] - precision * math.sin(a3),
                                p_n1[1] + precision * math.cos(a3)]  # 用a3计算的终点
                        a4 = angle(direc, p_n4[0], p_n4[1])  # 用a3计算的终点斜率
                        a_m = (a1 + 2 * a2 + 2 * a3 + a4) / 6  # 加权平均斜率
                        tab_trace_x.append(p_n1[0] - precision * math.sin(a_m))
                        tab_trace_y.append(p_n1[1] + precision * math.cos(a_m))
                    tab.append([tab_trace_x, tab_trace_y])
            return tab

        def euler(direc):
            tab = []
            if direc == 'h':
                for y_ in axe:
                    tab_trace_hx, tab_trace_hy = [-1.0], [y_]
                    while tab_trace_hx[-1] < t1:
                        p00, p10, p01, p11, t, s = find_sim_points(tab_trace_hx[-1], tab_trace_hy[-1])
                        angle_moy = angle_moyen('h', p00, p10, p01, p11, t, s)
                        tab_trace_hx.append(tab_trace_hx[-1] + precision * math.cos(angle_moy))
                        tab_trace_hy.append(tab_trace_hy[-1] + precision * math.sin(angle_moy))
                    tab.append([tab_trace_hx, tab_trace_hy])
            else:
                for x_ in axe:
                    tab_trace_vx, tab_trace_vy = [x_], [-1.0]
                    while tab_trace_vy[-1] < t1:
                        p00, p10, p01, p11, t, s = find_sim_points(tab_trace_vx[-1], tab_trace_vy[-1])
                        angle_moy = angle_moyen('v', p00, p10, p01, p11, t, s)
                        tab_trace_vx.append(tab_trace_vx[-1] - precision * math.sin(angle_moy))
                        tab_trace_vy.append(tab_trace_vy[-1] + precision * math.cos(angle_moy))
                    tab.append([tab_trace_vx, tab_trace_vy])
            return tab

        def milieux_indx(tab_x, semi):
            milieux = []
            flag = True  # x<semi
            for i in range(len(tab_x)):
                if flag:
                    if tab_x[i] >= semi:
                        flag = False
                        milieux.append(i)
                else:
                    if tab_x[i] < semi:
                        flag = True
                        milieux.append(i)
            return milieux

        def append_reci_inverse(tab_main, tab_x, tab_y, indice):
            tempx = tab_x[:indice][::-1]
            tempy = tab_y[:indice][::-1]
            tab_main.append([[-x_ for x_ in tempx], [-y_ for y_ in tempy]])

        if methode == "rk":
            if symetrique:
                semi = (t0 + t1) / 2
                demih = runge_kutta_demi('h', axe, "direct")
                demih0 = []
                demih1 = []
                tab_h = []
                for i in range(len(demih)):
                    trace_x, trace_y = demih[i]
                    trace_x_reci, trace_y_reci = demih[taille - i - 1]
                    milieux = milieux_indx(trace_x, semi)
                    milieux_reci = milieux_indx(trace_x_reci, semi)
                    lm = len(milieux)
                    lmr = len(milieux_reci)
                    ind_semi = milieux[int(lm / 2)]
                    ind_semi_reci = milieux_reci[int(lmr / 2)]
                    if lm == 1:
                        demih0.append([trace_x[:ind_semi + 1], trace_y[:ind_semi + 1]])
                        if lmr == 1:
                            append_reci_inverse(demih1, trace_x_reci, trace_y_reci, milieux_reci[0])
                        else:
                            ind_semi2_reci = milieux_reci[int(lmr / 2) + 1]
                            ind_semi3_reci = milieux_reci[int(lmr / 2) - 1]
                            cas = (np.abs([trace_y[ind_semi] + trace_y_reci[ind_semi_reci],
                                           trace_y[ind_semi] + trace_y_reci[ind_semi2_reci],
                                           trace_y[ind_semi] + trace_y_reci[ind_semi3_reci]])).argmin()
                            if cas == 0:
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi_reci)
                            elif cas == 1:
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi2_reci)
                            else:
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi3_reci)
                    else:
                        ind_semi2 = milieux[int(lm / 2) + 1]
                        ind_semi3 = milieux[int(lm / 2) - 1]
                        if lmr == 1:
                            append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi_reci)
                            cas = (np.abs([trace_y[ind_semi] + trace_y_reci[ind_semi_reci],
                                           trace_y[ind_semi2] + trace_y_reci[ind_semi_reci],
                                           trace_y[ind_semi3] + trace_y_reci[ind_semi_reci]])).argmin()
                            if cas == 0:
                                demih0.append([trace_x[:ind_semi + 1], trace_y[:ind_semi + 1]])
                            elif cas == 1:
                                demih0.append([trace_x[:ind_semi2 + 1], trace_y[:ind_semi2 + 1]])
                            else:
                                demih0.append([trace_x[:ind_semi3 + 1], trace_y[:ind_semi3 + 1]])
                        else:
                            ind_semi2_reci = milieux_reci[int(lmr / 2) + 1]
                            ind_semi3_reci = milieux_reci[int(lmr / 2) - 1]
                            cas = (np.abs([trace_y[ind_semi] + trace_y_reci[ind_semi_reci],
                                           trace_y[ind_semi] + trace_y_reci[ind_semi2_reci],
                                           trace_y[ind_semi] + trace_y_reci[ind_semi3_reci],
                                           trace_y[ind_semi2] + trace_y_reci[ind_semi_reci],
                                           trace_y[ind_semi2] + trace_y_reci[ind_semi2_reci],
                                           trace_y[ind_semi2] + trace_y_reci[ind_semi3_reci],
                                           trace_y[ind_semi3] + trace_y_reci[ind_semi_reci],
                                           trace_y[ind_semi3] + trace_y_reci[ind_semi2_reci],
                                           trace_y[ind_semi3] + trace_y_reci[ind_semi3_reci]
                                           ])).argmin()
                            if cas == 0:
                                demih0.append([trace_x[:ind_semi + 1], trace_y[:ind_semi + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi_reci)
                            elif cas == 1:
                                demih0.append([trace_x[:ind_semi + 1], trace_y[:ind_semi + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi2_reci)
                            elif cas == 2:
                                demih0.append([trace_x[:ind_semi + 1], trace_y[:ind_semi + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi3_reci)
                            elif cas == 3:
                                demih0.append([trace_x[:ind_semi2 + 1], trace_y[:ind_semi2 + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi_reci)
                            elif cas == 4:
                                demih0.append([trace_x[:ind_semi2 + 1], trace_y[:ind_semi2 + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi2_reci)
                            elif cas == 5:
                                demih0.append([trace_x[:ind_semi2 + 1], trace_y[:ind_semi2 + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi3_reci)
                            elif cas == 6:
                                demih0.append([trace_x[:ind_semi3 + 1], trace_y[:ind_semi3 + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi_reci)
                            elif cas == 7:
                                demih0.append([trace_x[:ind_semi3 + 1], trace_y[:ind_semi3 + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi2_reci)
                            else:
                                demih0.append([trace_x[:ind_semi3 + 1], trace_y[:ind_semi3 + 1]])
                                append_reci_inverse(demih1, trace_x_reci, trace_y_reci, ind_semi3_reci)
                for i in range(len(demih0)):
                    tab_h.append([demih0[i][0] + demih1[i][0], demih0[i][1] + demih1[i][1]])
                for i in range(len(tab_h)):
                    print(len(tab_h[i][0]) - (len(demih0[i][0]) + len(demih0[taille - i - 1][0]) / 2))
                """
                tab_milieux_h, tab_milieux_v = [], []
                for courbe in demih:
                    trace_x, trace_y = courbe
                    milieux = []
                    flag = True  # x<semi
                    for i in range(len(trace_x)):
                        if flag:
                            if trace_x[i] >= semi:
                                flag = False
                                milieux.append(i)
                        else:
                            if trace_x[i] < semi:
                                flag = True
                                milieux.append(i)
                    ind_semi = milieux[int(len(milieux) / 2)]
                    trace_x_demi, trace_y_demi = trace_x[:ind_semi + 1], trace_y[:ind_semi + 1]
                    trace_x_demi2 = -np.array(trace_x[:ind_semi][::-1])
                    trace_y_demi2 = -np.array(trace_y[:ind_semi][::-1])
                    demih0.append([trace_x_demi, trace_y_demi])
                    demih1.append([list(trace_x_demi2), list(trace_y_demi2)])
                    tab_milieux_h.append(milieux)
                demih1.reverse()

                for i in range(taille):
                    # demih1[i][0].reverse()
                    # demih1[i][1].reverse()
                    milieux = tab_milieux_h[i]
                    ind_semi2 = milieux[int(len(milieux) / 2) + 1]
                    if abs(demih0[i][1] - demih1[i][1]) <= abs(demih0[i][1] - demih1[ind_semi2][1]):
                        tab_h.append([demih0[i][0] + demih1[i][0], demih0[i][1] + demih1[i][1]])
                    else:
                        tab_h.append([demih0[i][0] + demih1[i][0], demih0[i][1] + demih1[i][1]])
                """
                res.append(tab_h)
                res.append([])
            else:
                res.append(runge_kutta('h'))
                res.append(runge_kutta('v'))
        else:
            res.append(euler('h'))
            res.append(euler('v'))
        return res

    def corriger(self, tab_trace, expr=None, symbol=None):
        sym = symbol if symbol is not None else sp.Symbol('x')
        cor_sym = expr if expr is not None else (1 / 2 * (1 - sp.exp(-sym)))
        cor = sp.lambdify(sym, cor_sym, "numpy")
        tab_cor_h, tab_cor_v = tab_trace
        for ligne in tab_cor_h:
            tab_x, tab_y = ligne
            i = 0
            while tab_x[i] < 0.8 * self.__t1:
                i += 1
            for j in range(i, len(tab_x)):
                tab_y[j] = cor(tab_x[i]) * tab_y[j]
        return [tab_cor_h, tab_cor_v]

    """Affichage"""

    def draw(self, direction='a', snb=None, mode="direct", display=True):
        """
        Afficher le diffeomorphisme par une image en 2D
        :param direction: soit 'h' pour la direction horientale, soit 'v' pour la direction verticale, soit l'autre pour
         tous afficher en une meme image
        :param snb:
        :param mode:
        :param display:
        :return:
        """
        taille = snb or self.snb
        tab_x, tab_y = self.tab_f(taille) if mode == "direct" else self.tab_f_points_reci(taille)
        if direction == 'h':
            plt.title("Diffeomorphisme dans la direction horientale")
            plt.plot(tab_x.T, tab_y.T)
        elif direction == 'v':
            plt.title("Diffeomorphisme dans la direction verticale")
            plt.plot(tab_x, tab_y)
        else:
            plt.title("Diffeomorphisme")
            plt.plot(tab_x.T, tab_y.T)
            plt.plot(tab_x, tab_y)
        if display:
            plt.show()

    def draw_df(self, direction='a', snb=None, mode="direct", display=True):
        """
        Afficher le champ de vecteurs pour un diffeomorphisme, les autres sont parailles que draw
        :param direction:
        :param snb:
        :param mode:
        :param display:
        :return:
        """
        taille = snb or self.snb
        tab_x, tab_y = self.tab_f(taille) if mode == "direct" else self.tab_f_points_reci(taille)
        tab_df = self.tab_df(taille) if mode == "direct" else self.tab_df_points_reci(taille)
        if direction == 'h':
            plt.quiver(tab_x, tab_y, tab_df[0][0], tab_df[1][0])
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.title("Champ de vecteurs horientals")
        elif direction == 'v':
            plt.quiver(tab_x, tab_y, tab_df[0][1], tab_df[1][1])
            plt.xlabel(r'$y_1$')
            plt.ylabel(r'$y_2$')
            plt.title("Champ de vecteurs verticals")
        else:
            plt.quiver(tab_x, tab_y, tab_df[0][0], tab_df[1][0])
            plt.quiver(tab_x, tab_y, tab_df[0][1], tab_df[1][1])
            plt.xlabel(r'$x_1$ et $y_1$')
            plt.ylabel(r'$x_2$ et $y_2$')
            plt.title("Champ de vecteurs")
        if display:
            plt.show()

    def draw_all(self, direction='a', snb=None, mode="direct", display=True):
        """
        Pour un diffeomorphisme, afficher une fois lui-meme et son champ de vecteurs en une figure, les aures sont
        parailles que draw et que draw_df
        :param direction:
        :param snb:
        :param mode:
        :param display:
        :return:
        """
        taille = snb or self.snb
        if direction == 'h':
            self.draw('h', taille, "direct", False)
            title = "Diffeomorphisme et son champ de vecteurs dans le sens horiental"
        elif direction == 'v':
            self.draw('v', taille, "direct", False)
            title = "Diffeomorphisme et son champ de vecteurs dans le sens vertical"
        else:
            self.draw('a', taille, "direct", False)
            title = "Diffeomorphisme et son champ de vecteurs"
        self.draw_df(direction, taille, mode, False)
        plt.title(title)
        if display:
            plt.show()

    def draw_angles_ligne(self, direction, snb=None, indice=None, val_min=None, val_max=None, display=True):
        taille = snb or self.snb
        if direction == 'h':
            case = 0
            direction_str = "ligne"
        else:
            case = 1
            direction_str = "colonne"
        ind = indice
        if ind is None:
            ind = taille // 2
        v_min, v_max = val_min, val_max
        tick = 0.25 * math.pi
        tab = ex.tab_angles_R(taille)[case][ind]
        if v_min is None:
            v_min = (min(tab) // tick - 1) * tick
        if v_max is None:
            v_max = (max(tab) // tick + 2) * tick
        axe = np.linspace(self.__t0, self.__t1, taille)
        plt.title("Angles de la ${}-ieme$ {} sur {} en total".format(ind, direction_str, taille))
        my_y_ticks = np.arange(v_min, v_max, tick)
        plt.yticks(my_y_ticks)
        plt.xlabel("x")
        plt.ylabel('$\Theta$')
        res = plt.plot(axe, tab)
        if display:
            plt.show()
        return res

    def play_angles(self, direction, snb=None, bsave=True, save_name=None):
        taille = snb or self.snb
        fig = plt.figure()
        if direction == 'h':
            case = 0
        else:
            case = 1
        tab = np.array(self.tab_angles_R(taille)[case])
        tick = 0.25 * math.pi
        val_min = (tab.min() // tick - 1) * tick
        val_max = (tab.max() // tick + 2) * tick
        tab_fig = []
        for i in range(taille):
            tab_fig.append(self.draw_angles_ligne(direction, taille, i, val_min, val_max, False))
        im_ani = anime.ArtistAnimation(fig, tab_fig, interval=50, repeat_delay=3000, blit=True)
        if bsave:
            name = save_name
            if name is None:
                name = "animation"
            im_ani.save(name + ".html")
        return im_ani

    def draw_trace(self, temps, direction='a', snb=None, multi=10, display=True, bcorrige=True, bsave=False,
                   save_name=None, methode="rk", symetric=False):
        taille = snb or self.snb
        trace_h, trace_v = self.trace(temps, taille, multi, methode=methode, symetrique=symetric)
        if bcorrige:
            trace_h, trace_v = self.corriger([trace_h, trace_v])
        if direction == 'h' or direction == 'a':
            for ligne in trace_h:
                plt.plot(ligne[0], ligne[1])
        if direction == 'v' or direction == 'a':
            for ligne in trace_v:
                plt.plot(ligne[0], ligne[1])
        if bsave:
            name = save_name
            if name is None:
                name = "trace.png"
            plt.savefig(name)
            plt.clf()
        if display:
            plt.show()

    def play(self, nb_frame, direction='a', snb=None, bsave=True, save_name=None):
        taille = snb or self.snb
        fig = plt.figure()
        tab_fig = []
        for i in range(0, 1, nb_frame):
            plt.clf()
            self.draw_trace(i, direction, taille, display=False)
            tab_fig.append(plt.gcf())
        im_ani = anime.ArtistAnimation(fig, tab_fig, interval=50, repeat_delay=3000, blit=True)
        if bsave:
            name = save_name
            if name is None:
                name = "animation"
            im_ani.save(name + ".html")
        return im_ani

    """Getter"""

    def gfsym(self):
        return self.__expr

    def gfnum(self):
        return self.__num

    def f(self, x_num, y_num):
        return self.__num(x_num, y_num)

    def df(self, x_num, y_num):
        """
        C'est la version de fonction python du differentiel du diffeomorphisme, qui prend en argument x_num et y_num,
        et qui retourne le differentiel (matrice jacobienne) dans ce point.
        :param x_num:
        :param y_num:
        :return:
        """
        return self.__df_num(x_num, y_num)


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
expression = f_ex2(0.2, 5, 5 * math.pi)[0]

# expr = x + 0.2 * sp.exp(-15 * (x ** 2 + y ** 2)), y + 0.045 * sp.exp(-10 * (x ** 2 + y ** 2))
ex = DiffeoInfini(expression)
ex.load_points_reci("tab_inverse.npy", le_t0, le_t1)
print(ex.gfnum())
# print(ex.f(0, 0))
# print(ex.gfsym())
# print(ex.df(0, 0))
ex.draw()
# ex.draw(mode="reci")
# ex.draw('h')
# ex.draw('v')
# print(ex.tab_df())
# ex.draw_df()
# ex.draw_df('h')
# ex.draw_df('v')
# ex.draw('h', display=False)
# ex.draw_df('h', mode="reci")
# ex.draw('v', display=False)
# ex.draw_df('v', mode="reci")
# ex.draw_all()
# ex.draw_all('h')
# ex.draw_all('v')
# print(ex.tab_df(la_taille))
# print(ex.tab_angles_R(la_taille))
# ex.draw_angles_ligne('h', taille=la_taille, indice=la_taille // 4, val_min=-0.01, val_max=0.01)
# ex.draw_angles_ligne('v', taille=la_taille, indice=la_taille // 4, val_min=-0.01, val_max=0.01)
# ani = ex.play_angles('h', bsave=True, save_name="angles_reci")
plt.title("Tracage par la methode de Runge-Kutta")
ex.draw_trace(1, bcorrige=False, symetric=False)
plt.title("Tracage par la methode d'Euler")
ex.draw_trace(1, bcorrige=False, methode="euler", symetric=False)
# ex.draw_trace(0.8, bcorrige=False, symetric=True)
"""
nb = 0
for i in np.linspace(0, 1, 60):
    ex.draw_trace(i, display=False, bcorrige=False, bsave=True, save_name=str(nb) + ".png",symetric=True)
    nb += 1
"""
