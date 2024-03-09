class Lattice:
    BASIS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, 1, 0), (1, -1, 0),
             (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
             (0, -1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1),
             (-1, -1, -1))

    def __init__(self, path_in='input.txt', path_out='output.txt'):
        self.path_in = path_in
        self.path_out = path_out
        self.Nz = None
        self.Ny = None
        self.Nx = None
        self.num_of_nodes = None
        self.minmaxcoord = None
        self.incr = None
        self.border_type = None

    def count_nods_xyz(self):
        """Создаём список (по осям) списков координат всех узлов
        решётки для дальнейших проверок и вычислений
        КОРРЕКТИРОВКА: упростил пока временно до вычисления количества узлов по всем осям"""
        self.num_of_nodes = []
        for n in range(6):  # вычисляем и складываем количество узлов
            # по разные стороны центра фигуры границы, для каждой оси.
            # На данный момент узел решётки совпадает с центром фигуры границы
            # и откладывается от него до границ области вычислений.
            # При этом если шаги узлов решётки кратны расстояниям, получаем много узлов решётки,
            # расположенных ровно в границе области вычислений! Уточнить нормально ли это!!!!!
            if not n % 2:
                self.num_of_nodes.append(
                    int(abs((self.minmaxcoord[n] - self.figure_centre[n // 2]) / self.incr[n // 2])))
            else:
                self.num_of_nodes.append(
                    int(abs((self.minmaxcoord[n] - self.figure_centre[n // 2]) / self.incr[n // 2])))
        self.Nx = self.num_of_nodes[0] + self.num_of_nodes[1] + 1
        self.Ny = self.num_of_nodes[2] + self.num_of_nodes[3] + 1
        self.Nz = self.num_of_nodes[4] + self.num_of_nodes[5] + 1

    @staticmethod
    def vect_sum(v1: tuple, v2: tuple) -> tuple:
        """Возвращает сумму двух векторов"""
        return v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]

    @staticmethod
    def vect_num_mult(n: float, v: tuple) -> tuple:
        """Возвращает произведение вектора на число"""
        return v[0] * n, v[1] * n, v[2] * n

    @staticmethod
    def vect_scalar_mult(v1: tuple, v2: tuple) -> float:
        """Возвращает скалярное произведение векторов"""
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

    @staticmethod
    def sign_of_num(x: float):
        return 0 if abs(x) == 0 else int(x / abs(x))

    # def isoutborder(self, x, y, z):
    #     if x == self.x_min:
    #         self.bordernum = 2
    #         return True
    #     elif x == self.x_min + (self.Nx - 1) * self.incr[0]:
    #         self.bordernum = 3
    #         return True
    #     elif y == self.y_min:
    #         self.bordernum = 4
    #         return True
    #     elif y == self.y_min + (self.Ny - 1) * self.incr[1]:
    #         self.bordernum = 5
    #         return True
    #     elif z == self.z_min:
    #         self.bordernum = 6
    #         return True
    #     elif z == self.z_min + (self.Nz - 1) * self.incr[2]:
    #         self.bordernum = 7
    #         return True
    #     else:
    #         return False


class Figure:
    pass


class Sphere(Figure):
    def __init__(self, centre=None, size=None):
        self.centre = centre
        self.size = size
    def isincheck(self, o: tuple):
        # o - координаты узла
        """Определаяем внешний/внутренний ли узел по
        его координатам и координатам центра сферы и её радиуса
        True, если узел внутренний (за пределами границы, внутри расчётной области)"""
        return ((o[0] - self.centre[0]) ** 2 + (o[1] - self.centre[1]) ** 2 + (
                o[2] - self.centre[2]) ** 2) >= self.size ** 2

    def isbordercheck(self, o: tuple):
        # o - координаты узла
        """Определаяем, граничит ли внешний узел с границей сферы (а значит с внутренним узлом)
        хотя бы по одной своей координате; определяем по его координатам и координатам
        центра сферы и её радиуса
        True, если узел граничит со сферой"""
        for vector in Lattice.BASIS:
            ans = self.isincheck((o[0] + self.incr[0] * vector[0], o[1] + self.incr[1] * vector[1],
                                  o[2] + self.incr[2] * vector[2]))
            if ans:
                return True
        return False

    def nearbordercheck(self, o: tuple, u: tuple) -> float:
        # o - координаты узла, u - базисный вектор
        """Определаяем, граничит ли данный узел с границей сферы
         в пределах и направлении заданного вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел не граничит в этом направлении, или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению вектора)
        в пределах расстояния между узлами в этом направлении."""
        sph = self.centre  # координаты центра сферы
        r = self.size  # радиус сферы
        if ((o[0] - sph[0]) ** 2 + (o[1] - sph[1]) ** 2 + (o[2] - sph[2]) ** 2) ** 0.5 == r:
            return 0
        ans1 = self.isincheck(o)
        ans2 = self.isincheck((o[0] + u[0] * self.incr[0],
                               o[1] + u[1] * self.incr[1], o[2] + u[2] * self.incr[2]))
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0

        o_sph = self.vect_sum(o, self.vect_num_mult(-1, sph))
        a = self.vect_scalar_mult(u, u)
        b = 2 * self.vect_scalar_mult(u, o_sph)
        c = self.vect_scalar_mult(o_sph, o_sph) - r ** 2
        det = (b ** 2 - 4 * a * c)
        if det >= 0:
            d1 = (- b + det ** 0.5) / (2 * a)
            d2 = (- b - det ** 0.5) / (2 * a)
            if d1 >= 0 and d2 >= 0:
                return min(d1 * (a ** 0.5), d2 * (a ** 0.5))
            elif d1 >= 0 or d2 >= 0:
                return max(d1 * (a ** 0.5), d2 * (a ** 0.5))
            else:
                return 0
        else:
            return 0

    def print_normal_and_distance(self, n, o: tuple):
        # o - координаты узла
        """Вычисляет вектор нормали к поверхности для сферы, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        dx = self.centre[0] - o[0]
        dy = self.centre[1] - o[1]
        dz = self.centre[2] - o[2]
        length = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        if length - self.size == 0:  # если узел лежит ровно на сфере
            # возвращаем нулевые вектор нормали и расстояние:
            return 0, 0, 0, 0
        else:
            match n:
                case 0 | 2 | 3 | 4 | 5 | 6 | 7:  # внутренние узлы (снаружи сферы)
                    # возвращаем вектор нормали и расстояние:
                    return (dx / length), (dy / length), (dz / length), (length - self.size)
                case 1:  # граничные узлы (внутри сферы)
                    # возвращаем вектор нормали и расстояние:
                    return (-dx / length), (-dy / length), (-dz / length), (self.size - length)


class Ellipsoid(Lattice):
    def __init__(self, path_in='input.txt', path_out='output.txt'):
        super().__init__(path_in, path_out)

    pass


class Cube(Lattice):
    def __init__(self, path_in='input.txt', path_out='output.txt'):
        super().__init__(path_in, path_out)

    def isincheck(self, o: tuple):
        # o - координаты узла
        """ Определаяем внутренний/внешний ли узел по
        его координатам и координатам центра куба и размеру его грани
        True, если узел внутренний (за пределами границы, внутри расчётной области)"""
        return (abs(o[0] - self.figure_centre[0]) >= self.figure_size[0] / 2 or
                abs(o[1] - self.figure_centre[0]) >= self.figure_size[1] / 2 or
                abs(o[2] - self.figure_centre[0]) >= self.figure_size[2] / 2)

    def isbordercheck(self, o: tuple):
        # o - координаты узла
        """Определаяем, граничит ли внешний узел с границей куба (а значит с внутренним узлом)
        хотя бы по одной своей координате;
        определяем по его координатам и координатам центра куба и его размеру
        True, если узел граничит с кубом"""
        for vector in self.BASIS:
            ans = self.isincheck((o[0] + self.incr[0] * vector[0], o[1] + self.incr[1] * vector[1],
                                  o[2] + self.incr[2] * vector[2]))
            if ans:
                return True
        return False

    def nearbordercheck(self, o: tuple, u: tuple) -> float:
        # o - координаты узла, u - базисный вектор
        """Определаяем, граничит ли данный узел с границей куба
         в пределах и направлении заданного вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел не граничит в этом направлении, или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению вектора)
        в пределах расстояния между узлами в этом направлении."""
        cb = self.figure_centre  # координаты центра куба
        a = self.figure_size  # размер куба
        if abs(o[0] - cb[0]) == a / 2 or abs(o[1] - cb[1]) == a / 2 or abs(o[2] - cb[2]) == a / 2:
            return 0

        ans1 = self.isincheck(o)
        ans2 = self.isincheck((o[0] + u[0] * self.incr[0],
                               o[1] + u[1] * self.incr[1], o[2] + u[2] * self.incr[2]))
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0

        # вычисляем по скольки координатам точка граничит с кубом в пределах узла
        # и расстояние до границы:
        numoftrue = 0
        delta = [abs(abs(o[n] - cb[n]) - a / 2) for n in range(3)]
        ans = [False for _ in range(3)]
        for n in range(3):
            if delta[n] <= self.incr[n]:
                numoftrue += 1
                ans[n] = True
        # вычисляем расстояние до куба для разных вариантов расположения узла:
        xoryorz_dist = 0  # сначала находим расстояние до границы по одной из координат, которая определяет
        # длину всего вектора до границы для конкретного расположения узла относительно куба
        match numoftrue:
            case 1:
                xoryorz_dist = ans[0] * delta[0] + ans[1] * delta[1] + ans[2] * delta[2]
            case 2:
                if ans1:  # если снаружи куба
                    xoryorz_dist = max(ans[0] * delta[0], ans[1] * delta[1], ans[2] * delta[2])
                elif ans2:  # если внутри куба
                    # находим, какие из компонент (по осям) базисных векторов
                    # смотрят от этой точки в сторону границы куба по этой оси (ближайшей в пределах шага):
                    dist_by_axis = []
                    for n in range(3):
                        chek = ans[n] * (self.sign_of_num(u[n]) == self.sign_of_num(o[n] - cb[n]))
                        if chek:
                            dist_by_axis.append(delta[n])
                    xoryorz_dist = dist_by_axis[0] if len(dist_by_axis) == 1 else min(dist_by_axis)
            case 3:
                if ans1:  # если снаружи куба
                    xoryorz_dist = max(delta)
                elif ans2:  # если внутри куба
                    # находим, какие из компонент (по осям) базисных векторов
                    # смотрят от этой точки в сторону границы куба по этой оси (ближайшей в пределах шага):
                    dist_by_axis = []
                    for n in range(3):
                        chek = self.sign_of_num(u[n]) == self.sign_of_num(o[n] - cb[n])
                        if chek:
                            dist_by_axis.append(delta[n])
                    xoryorz_dist = dist_by_axis[0] if len(dist_by_axis) == 1 else min(dist_by_axis)
        distance = xoryorz_dist * (self.vect_scalar_mult(u, u) ** 0.5)
        return distance

    def print_normal_and_distance(self, n, o):
        """Вычисляет вектор нормали к поверхности для куба, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        dx = self.figure_centre[0] - o[0]
        dy = self.figure_centre[1] - o[1]
        dz = self.figure_centre[2] - o[2]
        a = self.figure_size
        ans1 = abs(dx) == a / 2
        ans2 = abs(dy) == a / 2
        ans3 = abs(dz) == a / 2
        ans4 = abs(dx) <= a / 2
        ans5 = abs(dy) <= a / 2
        ans6 = abs(dz) <= a / 2
        if (ans1 and ans5 and ans6) or (ans2 and ans4 and ans6) or (ans3 and ans4 and ans5):
            # возвращаем вектор нормали и расстояние:
            return 0, 0, 0, 0
        else:
            norm_x, norm_y, norm_z, length = 0, 0, 0, 1
            match n:
                case 0 | 2 | 3 | 4 | 5 | 6 | 7:  # внутренние узлы (снаружи куба)
                    # ищем вектор нормали к ближайшей границе (единичный) и расстояние
                    # для разных положений точки относительно куба:
                    if ans5 and ans6:
                        length = abs(dx) - a / 2
                        norm_x = dx / abs(dx)
                        norm_y = 0
                        norm_z = 0
                    elif ans4 and ans6:
                        length = abs(dy) - a / 2
                        norm_x = 0
                        norm_y = dy / abs(dy)
                        norm_z = 0
                    elif ans4 and ans5:
                        length = abs(dz) - a / 2
                        norm_x = 0
                        norm_y = 0
                        norm_z = dz / abs(dz)
                    elif ans4 and not ans5 and not ans6:
                        length = ((abs(dy) - a / 2) ** 2 + (abs(dz) - a / 2) ** 2) ** 0.5
                        norm_x = 0
                        norm_y = (dy - self.sign_of_num(dy) * a / 2) / length
                        norm_z = (dz - self.sign_of_num(dz) * a / 2) / length
                    elif ans5 and not ans4 and not ans6:
                        length = ((abs(dx) - a / 2) ** 2 + (abs(dz) - a / 2) ** 2) ** 0.5
                        norm_x = (dx - self.sign_of_num(dx) * a / 2) / length
                        norm_y = 0
                        norm_z = (dz - self.sign_of_num(dz) * a / 2) / length
                    elif ans6 and not ans4 and not ans5:
                        length = ((abs(dx) - a / 2) ** 2 + (abs(dy) - a / 2) ** 2) ** 0.5
                        norm_x = (dx - self.sign_of_num(dx) * a / 2) / length
                        norm_y = (dy - self.sign_of_num(dy) * a / 2) / length
                        norm_z = 0
                    elif not ans4 and not ans5 and not ans6:
                        length = ((abs(dx) - a / 2) ** 2 + (abs(dy) - a / 2) ** 2 + (abs(dz) - a / 2) ** 2) ** 0.5
                        norm_x = (dx - self.sign_of_num(dx) * a / 2) / length
                        norm_y = (dy - self.sign_of_num(dy) * a / 2) / length
                        norm_z = (dz - self.sign_of_num(dz) * a / 2) / length
                    # возвращаем вектор нормали и расстояние:
                    return norm_x, norm_y, norm_z, length
                case 1:  # граничные узлы (внутри куба)
                    dx *= -1
                    dy *= -1
                    dz *= -1
                    an1 = abs(dx) >= abs(dy)
                    an2 = abs(dx) >= abs(dz)
                    an3 = abs(dy) >= abs(dx)
                    an4 = abs(dy) >= abs(dz)
                    an5 = abs(dz) >= abs(dx)
                    an6 = abs(dz) >= abs(dy)
                    if an1 and an2:
                        length = a / 2 - abs(dx)
                        norm_x = dx / abs(dx)
                        norm_y = 0
                        norm_z = 0
                    elif an3 and an4:
                        length = a / 2 - abs(dy)
                        norm_x = 0
                        norm_y = dy / abs(dy)
                        norm_z = 0
                    elif an5 and an6:
                        length = a / 2 - abs(dz)
                        norm_x = 0
                        norm_y = 0
                        norm_z = dz / abs(dz)
                    # возвращаем вектор нормали и расстояние:
                    return norm_x, norm_y, norm_z, length


class Parallelepiped(Lattice):
    def __init__(self, path_in='input.txt', path_out='output.txt'):
        super().__init__(path_in, path_out)

    pass


def get_input():
    """Достаём все параметры из входного файла"""
    try:
        with open("input.txt", encoding='utf-8') as file:
            lines = file.readlines()
            minmaxcoord = tuple(map(float, lines[0].split(';')))  # крайние
            # координаты (мин, макс) области вычислений по осям координат
            print('Крайние координаты (мин, макс) области вычислений: ', minmaxcoord)
            incr = tuple(map(float, lines[1].split(';')))  # шаги решётки по осям координат
            print('Шаги решётки по осям координат: ', incr)
            border_type = int(lines[2].strip())  # тип фигуры для задания границы
            # вытаскиваем параметры для каждого типа фигуры
            t = tuple(map(float, lines[3].split(';')))
            match border_type:
                case 0:  # произвольная граница с заданием по точкам/аналитически
                    pass
                case 1:  # сфера
                    figure_centre, figure_size = t[:-1], t[-1] / 2  # координаты центра сферы
                    # и радиус сферы
                    print(f'СФЕРА с центром {figure_centre} и радиусом {figure_size}')
                    lattice = Sphere()

                case 2:  # эллипсоид
                    figure_centre, figure_size = t[:-3], tuple(n / 2 for n in t[-3::])  # координаты центра эллипсоида
                    # и полуосей a, b, c
                    print(
                        f'ЭЛЛИПСОИД с центром {figure_centre} и полуосями {figure_size[0]}, {figure_size[1]},'
                        f' {figure_size[2]}')
                    lattice = Ellipsoid()
                case 3:  # куб
                    figure_centre, figure_size = t[:-1], t[-1]  # координаты центра куба
                    # и длина ребра куба
                    print(f'КУБ с центром {figure_centre} и ребром {figure_size}')
                    lattice = Cube()
                case 4:  # параллелепипед
                    figure_centre, figure_size = t[:-3], t[-3::]  # координаты центра параллелепипеда
                    # и длины рёбер a, b, c
                    print(
                        f'ПАРАЛЛЕЛЕПИПЕД с центром {figure_centre} и рёбрами {figure_size[0]}, {figure_size[1]},'
                        f' {figure_size[2]}')
                    lattice = Parallelepiped()

            lattice.minmaxcoord = minmaxcoord
            lattice.incr = incr
            lattice.border_type = border_type
            lattice.centre = figure_centre
            lattice.size = figure_size
    except Exception as error:
        print(f'Неверный формат входного файла:\n{error}')
    return lattice


def create_outfile():
    """Вызываем все функции, делаем вычисления и записываем в файл"""
    latt = get_input()
    latt.count_nods_xyz()
    latt.x_min = latt.centre[0] - latt.num_of_nodes[0] * latt.incr[0]
    latt.y_min = latt.centre[1] - latt.num_of_nodes[2] * latt.incr[1]
    latt.z_min = latt.centre[2] - latt.num_of_nodes[4] * latt.incr[2]
    try:
        with open(latt.path_out, "w") as file:
            file.write(
                f'{latt.x_min};{latt.y_min};{latt.z_min}\n')  # координаты
            # левого нижнего узла решётки (минимальные координаты узла по каждому направлению
            # из Ox, Oy, Oz)

            file.write(f'{latt.incr[0]};{latt.incr[1]};{latt.incr[2]}\n')  # шаг решётки по каждому
            # направлению
            file.write(f'{latt.Nx};{latt.Ny};{latt.Nz}\n')  # количество узлов по каждой оси

            # Количество внутренних узлов, и далее через ; количество узлов
            # для каждой из границ расчётной области
            # ТАКЖЕ в этом блоке сохраняются все внешние и граничные узлы в виде списка кортежей их координат,
            # и распределяются по этим группам
            nums = [0, 0, latt.Ny * latt.Nz, latt.Ny * latt.Nz, latt.Nx * latt.Nz, latt.Nx * latt.Nz,
                    latt.Nx * latt.Ny, latt.Nx * latt.Ny]
            latt.nods_to_write = [[] for _ in range(8)]
            for k in range(1, latt.Nz - 1):  # убираем из циклов крайние значения - узлы границ
                # расчётных областей, т.к. они отдельно ниже генерируются
                z = latt.z_min + k * latt.incr[2]
                for j in range(1, latt.Ny - 1):
                    y = latt.y_min + j * latt.incr[1]
                    for i in range(1, latt.Nx - 1):
                        x = latt.x_min + i * latt.incr[0]
                        if latt.isincheck((x, y, z)):
                            nums[0] += 1
                            latt.nods_to_write[0].append((i, j, k))
                        elif latt.isbordercheck((x, y, z)):
                            nums[1] += 1
                            latt.nods_to_write[1].append((i, j, k))
            # Количество внутренних узлов записываем:
            file.write(f'{nums[0]}')
            # и количество граничных узлов записываем:
            for i in range(1, 8):
                file.write(f';{nums[i]}')
            file.write('\n')

            # генерируем узлы границ расчётных областей
            for k in range(latt.Nz):
                for j in range(latt.Ny):
                    latt.nods_to_write[2].append((0, j, k))
                    latt.nods_to_write[3].append((latt.Nx - 1, j, k))
            for k in range(latt.Nz):
                for i in range(latt.Nx):
                    latt.nods_to_write[4].append((i, 0, k))
                    latt.nods_to_write[5].append((i, latt.Ny - 1, k))
            for j in range(latt.Ny):
                for i in range(latt.Nx):
                    latt.nods_to_write[6].append((i, j, 0))
                    latt.nods_to_write[7].append((i, j, latt.Nz - 1))

            # Далее всё повторяется для каждого последующего вектора:
            #     id; i; j; k;
            #     26 * расстояний (от узла до границы, в пределах одного узла) по 26 заданным направлениям;
            #     вектор нормали к ближайшей границе (единичный);
            #     расстояние до ближайшей границы (по вектору нормали будет).
            #
            #     Идентификатор узла определяется id = i + (j + k * Ny) * Nx,
            #     сначала список узлов внутренних, потом по очереди всех границ
            for n in range(8):
                # сначала для внутренних узлов считаем и записываем (n=0), затем для граничных узлов фигуры (n=1),
                # затем для границ расчётной области (n > 1):
                for i, j, k in latt.nods_to_write[n]:
                    id_ = i + (j + k * latt.Ny) * latt.Nx
                    file.write(f'{id_};{i};{j};{k}')
                    x = latt.x_min + i * latt.incr[0]
                    y = latt.y_min + j * latt.incr[1]
                    z = latt.z_min + k * latt.incr[2]
                    # 26 * расстояний по 26 заданным направлениям:
                    for vector in latt.BASIS:
                        file.write(f';{latt.nearbordercheck((x, y, z), vector)}')
                    # вектор нормали к ближайшей границе (единичный)
                    # и расстояние до ближайшей границы (по вектору нормали):
                    normal = latt.print_normal_and_distance(n, (x, y, z))
                    file.write(f';{normal[0]};{normal[1]};{normal[2]};{normal[3]}')
                    file.write(f'\n')
    except Exception as error:
        print(f'Ошибка при создании выходного файла:\n{error}')


create_outfile()
