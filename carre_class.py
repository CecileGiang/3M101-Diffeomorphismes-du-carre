import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import sympy as sp


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

    def __init__(self, expr, expr_reci=None, t0=-1, t1=1, vars_sym=(sp.symbols("x y"))):
        """
        Pour creer une instance d'un diffeomorphisme de I^2 a I^2, il faut donner son expression mathematique,
        c'est-a-dire, une expression symbolique, ou on represente ses deux variables par x et y par default, et il faut
        aussi preciser le domaine I^2 sous forme [t0, t1]^2. De plus, pour accederer le calcul de sa fonction reciproque
        on peut se donner son expression symbolique comme une option.
        :param expr: l'expression symbolique du diffeomorphisme
        :param expr_reci: l'expression symbolique du diffeomorphisme reciproque
        :param t0: valeur minimale dans l'interval I
        :param t1: valeur maximale dans l'interval I
        :param vars_sym: les symboles qui representent les deux variables du diffeomorphisme
        """
        """
        Variables de base :
        expr: l'expression symbolique de ce diffeomorphisme
        expr_reci: l'expression symbolique du diffeomorphisme reciproque
        """
        self.expr = expr
        self.expr_reci = expr_reci
        self._t0, self._t1 = t0, t1
        self._x, self._y = vars_sym  # les representations symboliques de x et y
        self.num = sp.lambdify((self._x, self._y), self.expr, "numpy")  # la fonction de python correspondant a _expr
        # la fonction de python correspondant a _expr_reci
        self.num_reci = sp.lambdify((self._x, self._y), self.expr_reci, "numpy") if self.expr_reci is not None else None
        self.df_sym = self._df_sym()  # l'expression symbolique du differentiel de ce diffeomorphisme
        self.df_num = sp.lambdify((self._x, self._y), self.df_sym,
                                  "numpy")  # la fonction de python correspondant a _df_sym
        self._plan = None  # meshes grid d'un plan, contenant meshgrid
        self._tab_f = None  # f(plan)
        self._tab_df = None  # tab_df(plan)
        # Les points dans l'ensemble depart dont images par ce diffeomorphisme sont au meshgrid de l'ensemble arrive
        # t0, t1, taille, tableau en 3 dimentions (ligne,colonne,(x,y))
        self._tab_points_reci = None
        self._tab_f_points_reci = None
        self._tab_df_points_reci = None
        self._tab_angles_R = None
        self.simulation = None

    def change_domain(self, t0=None, t1=None):
        flag = False
        if t0 is not None and t0 != self._t0:
            self._t0 = t0
            flag = True
        if t1 is not None and t1 != self._t1:
            self._t1 = t1
            flag = True
        if flag:
            self._plan = None
            self._tab_points_reci = None
            self._tab_f = None
            self._tab_df = None
            self._tab_angles_R = None
        return flag

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
        if self.change_domain(t0, t1) or self._plan is None or taille != len(self._plan):
            self._plan = np.meshgrid(np.linspace(t0, t1, taille), np.linspace(t0, t1, taille))
        return self._plan

    def f(self, x_num, y_num):
        return self.num(x_num, y_num)

    def tab_f(self, t0=-1, t1=1, taille=50):
        if self._tab_f is None or taille != len(self._tab_f[0]):
            axe_x, axe_y = self.plan(t0, t1, taille)
            tab_x, tab_y = self.f(axe_x, axe_y)
            self._tab_f = (tab_x, tab_y)
        return self._tab_f

    def _df_sym(self):
        """
        对一个符号表达函数f：I^2->I^2求其微分表达式。注意返回的矩阵是一个一维列表，先从上到下，再从左到右
        Pour une fonction symbolique f：I^2->I^2, on calcul son différentiel.
        Attention, le résultat est un liste en 1 dimensionla, en lisant la matrice de différentiel de haut à bas,
        et de gauche à droit
        :param _f:
        :return: | &φ1/&x  &φ1/&y |
                  | &φ2/&x  &φ2/&y |
        """
        df_sym = sp.Matrix([[sp.diff(self.expr[0], self._x), sp.diff(self.expr[0], self._y)],
                            [sp.diff(self.expr[1], self._x), sp.diff(self.expr[1], self._y)]])
        return df_sym

    def df(self, x_num, y_num):
        """
        C'est la version de fonction python du differentiel du diffeomorphisme, qui prend en argument x_num et y_num,
        et qui retourne le differentiel (matrice jacobienne) dans ce point.
        :param x_num:
        :param y_num:
        :return:
        """
        return self.df_num(x_num, y_num)

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
        if self._tab_df is None or taille != len(self._tab_df[0][0]):
            axe_x, axe_y = self.plan(t0, t1, taille)
            self._tab_df = self.df_num(axe_x, axe_y)
        return self._tab_df

    def f_1(self, x_num, y_num):
        if self.num_reci is not None:
            return self.num_reci(x_num, y_num)
        else:
            return None

    def load_points_reci(self, path, t0, t1, struc="tab"):
        if t0 != self._t0 or t1 != self._t1:
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

    def tab_points_reci(self, t0=-1, t1=1, taille=50, multi=10):

        if self._tab_points_reci is None:
            if self.num_reci is None:
                pass
            axe_x, axe_y = self.plan(t0, t1, taille)
            tab_x, tab_y = self.num_reci(axe_x, axe_y)
            self._tab_points_reci = (tab_x, tab_y)

        return self._tab_points_reci

    def tab_f_points_reci(self, t0=-1, t1=1, taille=50, multi=10):
        if self._tab_f_points_reci is None or taille != len(self._tab_f_points_reci[0]):
            tab_reci_x, tab_reci_y = self.tab_points_reci(t0, t1, taille, multi)
            self._tab_f_points_reci = self.num(tab_reci_x, tab_reci_y)
        return self._tab_f_points_reci

    def tab_df_points_reci(self, t0=-1, t1=1, taille=50, multi=10):
        if self._tab_df_points_reci is None or taille != len(self._tab_df_points_reci[0][0]):
            tab_reci_x, tab_reci_y = self.tab_points_reci(t0, t1, taille, multi)
            self._tab_df_points_reci = self.df_num(tab_reci_x, tab_reci_y)
        return self._tab_df_points_reci

    def tab_angles_R(self, t0=-1, t1=1, taille=50, multi=10):
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

        if self._tab_angles_R is None or taille != len(self._tab_angles_R[0]):
            tab_df = self.tab_df_points_reci(t0, t1, taille, multi)
            tab_angles_x_2pi = np.arctan2(tab_df[1][0], tab_df[0][0])
            tab_angles_y_2pi = np.arctan2(tab_df[1][1], tab_df[0][1])

            tab_angles_x_R = corrigeur(tab_angles_x_2pi)
            tab_angles_y_R = corrigeur(tab_angles_y_2pi.T) - math.pi / 2

            # self._tab_angles_R[3] = np.array(tab_angles_x_R), np.array(tab_angles_y_R)
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

    def tab_inverse(self, t0=-1, t1=1, taille=50, multi=10):
        axe = np.linspace(t0, t1, taille)
        axe_x, axe_y = np.meshgrid(axe, axe)
        axe2 = np.linspace(t0, t1, taille * multi)
        axe2_x, axe2_y = np.meshgrid(axe2, axe2)
        f = self.num
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

    def trace(self, temps=1, t0=-1, t1=1, taille=50, multi=10, precision=0.005, methode="rk"):
        tab_angles = self.tab_angles_R(t0, t1, taille, multi) * temps
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

        def angle_moyen(direc, p00_, p10_, p01_, p11_, t_, s_):
            if direc == 'h':
                a00 = np.array([tab_angles[0, p00_[1], p00_[0]]])
                a10 = np.array([tab_angles[0, p10_[1], p10_[0]]])
                a01 = np.array([tab_angles[0, p01_[1], p01_[0]]])
                a11 = np.array([tab_angles[0, p11_[1], p11_[0]]])
            else:
                a00 = np.array([tab_angles[1, p00_[0], p00_[1]]])
                a10 = np.array([tab_angles[1, p10_[0], p10_[1]]])
                a01 = np.array([tab_angles[1, p01_[0], p01_[1]]])
                a11 = np.array([tab_angles[1, p11_[0], p11_[1]]])
            angle = (1 - t_) * ((1 - s_) * a00 + s_ * a01) + t_ * ((1 - s_) * a10 + s_ * a11)
            return angle

        def angle(direc, x_, y_):
            p00_, p10_, p01_, p11_, t_, s_ = find_sim_points(x_, y_)
            return angle_moyen(direc, p00_, p10_, p01_, p11_, t_, s_)

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

        if methode == "rk":
            res.append(runge_kutta('h'))
            res.append(runge_kutta('v'))
        else:
            res.append(euler('h'))
            res.append(euler('v'))
        self.simulation = res
        return self.simulation

    def draw(self, direction='a', t0=-1, t1=1, taille=50, mode="direct", display=True):
        """
        Afficher le diffeomorphisme par une image en 2D
        :param direction: soit 'h' pour la direction horientale, soit 'v' pour la direction verticale, soit l'autre pour
         tous afficher en une meme image
        :param t0:
        :param t1:
        :param taille:
        :param mode:
        :param display:
        :return:
        """
        tab_x, tab_y = self.tab_f(t0, t1, taille) if mode == "direct" else self.tab_f_points_reci(t0, t1, taille)
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

    def draw_df(self, direction='a', t0=-1, t1=1, taille=50, mode="direct", display=True):
        """
        Afficher le champ de vecteurs pour un diffeomorphisme, les autres sont parailles que draw
        :param direction:
        :param t0:
        :param t1:
        :param taille:
        :param mode:
        :param display:
        :return:
        """
        tab_x, tab_y = self.tab_f(t0, t1, taille) if mode == "direct" else self.tab_f_points_reci(t0, t1, taille)
        tab_df = self.tab_df(t0, t1, taille) if mode == "direct" else self.tab_df_points_reci(t0, t1, taille)
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

    def draw_all(self, direction='a', t0=-1, t1=1, taille=50, mode="direct", display=True):
        """
        Pour un diffeomorphisme, afficher une fois lui-meme et son champ de vecteurs en une figure, les aures sont
        parailles que draw et que draw_df
        :param direction:
        :param t0:
        :param t1:
        :param taille:
        :param mode:
        :param display:
        :return:
        """
        if direction == 'h':
            self.draw('h', t0, t1, taille, "direct", False)
            title = "Diffeomorphisme et son champ de vecteurs dans le sens horiental"
        elif direction == 'v':
            self.draw('v', t0, t1, taille, "direct", False)
            title = "Diffeomorphisme et son champ de vecteurs dans le sens vertical"
        else:
            self.draw('a', t0, t1, taille, "direct", False)
            title = "Diffeomorphisme et son champ de vecteurs"
        self.draw_df(direction, t0, t1, taille, mode, False)
        plt.title(title)
        if display:
            plt.show()

    def draw_angles_ligne(self, direction, t0=-1, t1=1, taille=50, indice=None, val_min=None, val_max=None,
                          display=True):
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
        tab = ex.tab_angles_R(t0, t1, taille)[case][ind]
        if v_min is None:
            v_min = (min(tab) // tick - 1) * tick
        if v_max is None:
            v_max = (max(tab) // tick + 2) * tick
        axe = np.linspace(t0, t1, taille)
        plt.title("Angles de la ${}-ieme$ {} sur {} en total".format(ind, direction_str, taille))
        my_y_ticks = np.arange(v_min, v_max, tick)
        plt.yticks(my_y_ticks)
        plt.xlabel("x")
        plt.ylabel('$\Theta$')
        res = plt.plot(axe, tab)
        if display:
            plt.show()
        return res

    def play_angles(self, direction, t0=-1, t1=1, taille=50, bsave=True, save_name=None):
        fig = plt.figure()
        if direction == 'h':
            case = 0
        else:
            case = 1
        tab = np.array(self.tab_angles_R(t0, t1, taille)[case])
        tick = 0.25 * math.pi
        val_min = (tab.min() // tick - 1) * tick
        val_max = (tab.max() // tick + 2) * tick
        tab_fig = []
        for i in range(taille):
            tab_fig.append(self.draw_angles_ligne(direction, t0, t1, taille, i, val_min, val_max, False))
        im_ani = anime.ArtistAnimation(fig, tab_fig, interval=50, repeat_delay=3000, blit=True)
        if bsave:
            name = save_name
            if name is None:
                name = "animation"
            im_ani.save(name + ".html")
        return im_ani

    def draw_trace(self, temps, direction='a', t0=-1, t1=1, taille=50, multi=10, display=True):
        trace_h, trace_v = self.trace(temps, t0, t1, taille, multi)
        if direction == 'h' or direction == 'a':
            for ligne in trace_h:
                plt.plot(ligne[0], ligne[1])
        if direction == 'v' or direction == 'a':
            for ligne in trace_v:
                plt.plot(ligne[0], ligne[1])
        if display:
            plt.show()

    def play(self, nb_frame, direction='a', t0=-1, t1=1, taille=50, bsave=True, save_name=None):
        fig = plt.figure()
        tab_fig = []
        for i in range(0, 1, nb_frame):
            plt.clf()
            self.draw_trace(i, direction, t0, t1, taille, display=False)
            tab_fig.append(plt.gcf())
        im_ani = anime.ArtistAnimation(fig, tab_fig, interval=50, repeat_delay=3000, blit=True)
        if bsave:
            name = save_name
            if name is None:
                name = "animation"
            im_ani.save(name + ".html")
        return im_ani


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
expr = f_ex2(0.2, 5, 5 * math.pi)[0]
# expr = x + 0.2 * sp.exp(-15 * (x ** 2 + y ** 2)), y + 0.045 * sp.exp(-10 * (x ** 2 + y ** 2))
ex = fonc_diff_infini(expr)
ex.load_points_reci("tab_inverse.npy", -1, 1)
# print(ex.num)
# print(ex.f(0, 0))
# print(ex.df_sym)
# print(ex.df(0, 0))
ex.draw()
# ex.draw(mode="reci")
# ex.draw('h')
# ex.draw('v')
# print(ex.tab_df())
# ex.draw_df()
# ex.draw_df('h')
# ex.draw_df('v')
ex.draw('h', display=False)
ex.draw_df('h', mode="reci")
ex.draw('v', display=False)
ex.draw_df('v', mode="reci")
# ex.draw_all()
# ex.draw_all('h')
# ex.draw_all('v')
# print(ex.tab_df(le_t0, le_t0, la_taille))
# print(ex.tab_angles_R(-le_t0, le_t1, la_taille))
# ex.draw_angles_ligne('h', taille=la_taille, indice=la_taille // 4, val_min=-0.01, val_max=0.01)
# ex.draw_angles_ligne('v', taille=la_taille, indice=la_taille // 4, val_min=-0.01, val_max=0.01)
# ani = ex.play_angles('h', bsave=True, save_name="angles_reci")
# tr = ex.trace(ex.tab_angles_R(le_t0, le_t1, la_taille), le_t0, le_t1, la_taille, 0.0001)
test = ex.trace()
ex.draw_trace(1.0)
ex.draw_trace(0.5)
# ex.play(30, save_name="trans")
# axe = np.linspace(-1, 1, 20)
# xx, yy = np.meshgrid(axe, axe)
# tab_distance = ex._distance(-1, -1, xx, yy)
# nn = math.ceil((20 // 2) * math.sqrt(2))
# resx, resy = ex._classifier_points_cles(tab_distance, xx, yy, 2 / 19, nn)
"""
for i in range(len(resx)):
    plt.scatter(resx[i], resy[i])
    plt.show()
"""
