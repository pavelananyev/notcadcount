class Point:
    def __init__(self, x, y=0, z=0):
        if type(x) in (int, float) and type(y) in (int, float) and type(z) in (int, float):
            self.x = x
            self.y = y
            self.z = z
        elif type(x) in (tuple, list):  # если передаём точку готовым кортежем или списком координат
            self.x = x[0]
            self.y = x[1]
            self.z = x[2]
        else:
            raise ValueError("Координаты должны быть числами!")

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Vector(x, y, z)


class Vector:
    def __init__(self, x=0, y=0, z=0):
        if type(x) in (int, float) and type(y) in (int, float) and type(z) in (int, float):
            self.x = x
            self.y = y
            self.z = z
        else:
            raise ValueError("Компоненты вектора должны быть числами!")

    def __add__(self, other):  # при сложении с точкой возвращает новую точку
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z) if type(other) is Point else Vector(x, y, z)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Vector(x, y, z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif type(other) in (int, float):
            return Vector(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Неподходящий тип данных для умножения на вектор")

    def __pow__(self, other):
        if type(other) is int and other > 0:
            out = 1
            for _ in range(other):
                out = self.__mul__(out)
            return out
        else:
            raise ValueError("Вектора можно возводить только в целую положительную степень")

    def square_of_norm(self):
        return self * self


class Lattice:
    BASIS = tuple(Vector(v[0], v[1], v[2]) for v in
                  ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, 1, 0),
                   (1, -1, 0),
                   (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
                   (0, -1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1),
                   (-1, -1, -1)))

    def __init__(self, path_in='input.txt', path_out='output.txt'):
        self.path_in = path_in
        self.path_out = path_out
        self.nods_to_write = None
        self.z_min = None
        self.y_min = None
        self.x_min = None
        self.Nz = None
        self.Ny = None
        self.Nx = None
        self.num_of_nodes = None
        self.minmaxcoord = None
        self.incr = None
        self.border_type = None
        self.figure = None

    @staticmethod
    def sign_of_num(x: float):
        return 0 if abs(x) == 0 else int(x / abs(x))

    def get_input(self):
        """Достаём все параметры из входного файла"""
        try:
            with open("input.txt", encoding='utf-8') as file:
                lines = file.readlines()
                self.minmaxcoord = tuple(map(float, lines[0].split(';')))  # крайние
                # координаты (мин, макс) области вычислений по осям координат
                print('Крайние координаты (мин, макс) области вычислений: ', self.minmaxcoord)
                self.incr = tuple(map(float, lines[1].split(';')))  # шаги решётки по осям координат
                print('Шаги решётки по осям координат: ', self.incr)
                self.border_type = int(lines[2].strip())  # тип фигуры для задания границы
                # вытаскиваем параметры для каждого типа фигуры
                t = tuple(map(float, lines[3].split(';')))
                match self.border_type:
                    case 0:  # произвольная граница с заданием по точкам/аналитически
                        pass
                    case 1:  # сфера
                        figure_centre, figure_size = t[:-1], t[-1] / 2  # координаты центра сферы
                        # и радиус сферы
                        print(f'СФЕРА с центром {figure_centre} и радиусом {figure_size}')
                        self.figure = Sphere(Point(figure_centre), figure_size, self)

                    case 2:  # эллипсоид
                        figure_centre, figure_size = t[:-3], tuple(
                            n / 2 for n in t[-3::])  # координаты центра эллипсоида
                        # и полуосей a, b, c
                        print(
                            f'ЭЛЛИПСОИД с центром {figure_centre} и полуосями {figure_size[0]}, {figure_size[1]},'
                            f' {figure_size[2]}')
                        self.figure = Ellipsoid(Point(figure_centre), figure_size, self)
                    case 3:  # куб
                        figure_centre, figure_size = t[:-1], t[-1]  # координаты центра куба
                        # и длина ребра куба
                        print(f'КУБ с центром {figure_centre} и ребром {figure_size}')
                        self.figure = Parallelepiped(Point(figure_centre), (figure_size, figure_size, figure_size),
                                                     self)
                    case 4:  # параллелепипед
                        figure_centre, figure_size = t[:-3], tuple(t[-3::])  # координаты центра параллелепипеда
                        # и длины рёбер a, b, c
                        print(
                            f'ПАРАЛЛЕЛЕПИПЕД с центром {figure_centre} и рёбрами {figure_size[0]}, {figure_size[1]},'
                            f' {figure_size[2]}')
                        self.figure = Parallelepiped(Point(figure_centre), figure_size, self)
        except Exception as error:
            print(f'Неверный формат входного файла:\n{error}')

    def count_nods_xyz(self):
        """Вычисление количества узлов по всем осям"""
        self.num_of_nodes = []
        for n in range(6):  # вычисляем и складываем количество узлов
            # по разные стороны центра фигуры границы, для каждой оси.
            # На данный момент узел решётки совпадает с центром фигуры границы
            # и откладывается от него до границ области вычислений.
            centre = (self.figure.centre.x, self.figure.centre.y, self.figure.centre.z)
            if not n % 2:
                self.num_of_nodes.append(
                    int(abs((self.minmaxcoord[n] - centre[n // 2]) / self.incr[n // 2])))
            else:
                self.num_of_nodes.append(
                    int(abs((self.minmaxcoord[n] - centre[n // 2]) / self.incr[n // 2])))
        self.Nx = self.num_of_nodes[0] + self.num_of_nodes[1] + 1
        self.Ny = self.num_of_nodes[2] + self.num_of_nodes[3] + 1
        self.Nz = self.num_of_nodes[4] + self.num_of_nodes[5] + 1

    def create_outfile(self):
        """Вызываем все функции, делаем вычисления и записываем в файл"""
        self.get_input()
        self.count_nods_xyz()
        self.x_min = self.figure.centre.x - self.num_of_nodes[0] * self.incr[0]
        self.y_min = self.figure.centre.y - self.num_of_nodes[2] * self.incr[1]
        self.z_min = self.figure.centre.z - self.num_of_nodes[4] * self.incr[2]
        try:
            with open(self.path_out, "w") as file:
                file.write(
                    f'{self.x_min};{self.y_min};{self.z_min}\n')  # координаты
                # левого нижнего узла решётки (минимальные координаты узла по каждому направлению
                # из Ox, Oy, Oz)

                file.write(f'{self.incr[0]};{self.incr[1]};{self.incr[2]}\n')  # шаг решётки по каждому
                # направлению
                file.write(f'{self.Nx};{self.Ny};{self.Nz}\n')  # количество узлов по каждой оси

                # Количество внутренних узлов, и далее через ; количество узлов
                # для каждой из границ расчётной области
                # ТАКЖЕ в этом блоке сохраняются все внешние и граничные узлы в виде списка кортежей их координат,
                # и распределяются по этим группам
                nums = [0, 0, self.Ny * self.Nz, self.Ny * self.Nz, self.Nx * self.Nz, self.Nx * self.Nz,
                        self.Nx * self.Ny, self.Nx * self.Ny]
                self.nods_to_write = [[] for _ in range(8)]
                for k in range(1, self.Nz - 1):  # убираем из циклов крайние значения - узлы границ
                    # расчётных областей, т.к. они отдельно ниже генерируются
                    z = self.z_min + k * self.incr[2]
                    for j in range(1, self.Ny - 1):
                        y = self.y_min + j * self.incr[1]
                        for i in range(1, self.Nx - 1):
                            x = self.x_min + i * self.incr[0]
                            if self.figure.isincheck(Point(x, y, z)):
                                nums[0] += 1
                                self.nods_to_write[0].append((i, j, k))
                            elif self.figure.isbordercheck(Point(x, y, z)):
                                nums[1] += 1
                                self.nods_to_write[1].append((i, j, k))
                # Количество внутренних узлов записываем:
                file.write(f'{nums[0]}')
                # и количество граничных узлов записываем:
                for i in range(1, 8):
                    file.write(f';{nums[i]}')
                file.write('\n')

                # генерируем узлы границ расчётных областей
                for k in range(self.Nz):
                    for j in range(self.Ny):
                        self.nods_to_write[2].append((0, j, k))
                        self.nods_to_write[3].append((self.Nx - 1, j, k))
                for k in range(self.Nz):
                    for i in range(self.Nx):
                        self.nods_to_write[4].append((i, 0, k))
                        self.nods_to_write[5].append((i, self.Ny - 1, k))
                for j in range(self.Ny):
                    for i in range(self.Nx):
                        self.nods_to_write[6].append((i, j, 0))
                        self.nods_to_write[7].append((i, j, self.Nz - 1))

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
                    for i, j, k in self.nods_to_write[n]:
                        id_ = i + (j + k * self.Ny) * self.Nx
                        file.write(f'{id_};{i};{j};{k}')
                        x = self.x_min + i * self.incr[0]
                        y = self.y_min + j * self.incr[1]
                        z = self.z_min + k * self.incr[2]
                        # 26 * расстояний по 26 заданным направлениям:
                        for vector in self.BASIS:
                            file.write(f';{self.figure.nearbordercheck(Point(x, y, z), vector)}')
                        # вектор нормали к ближайшей границе (единичный)
                        # и расстояние до ближайшей границы (по вектору нормали):
                        normal, distance = self.figure.print_normal_and_distance(Point(x, y, z))
                        file.write(f';{normal.x};{normal.y};{normal.z};{distance}')
                        file.write(f'\n')
        except Exception as error:
            print(f'Ошибка при создании выходного файла:\n{error}')


class Figure:
    def __init__(self, lattice: Lattice = None):
        self.lattice = lattice

    def isincheck(self, *args):
        raise NotImplementedError("В дочернем классе должен быть"
                                  "переопределён метод isincheck()")

    def isbordercheck(self, nod: Point):
        # nod - координаты внешнего узла (вне расчётной области, внутри поверхности)
        """Определаяем, граничит ли внешний узел с границей поверхности
        (а значит с внутренним узлом) хотя бы по одному базисному вектору в пределах шага решётки;
        True, если узел граничит с поверхностью"""
        for v in self.lattice.BASIS:
            ans = self.isincheck(Point(nod.x + v.x * self.lattice.incr[0],
                                       nod.y + v.y * self.lattice.incr[1],
                                       nod.z + v.z * self.lattice.incr[2]))
            if ans:
                return True
        return False


class Sphere(Figure):
    def __init__(self, centre=Point(0, 0, 0), size=None, lattice: Lattice = None):
        super().__init__(lattice)
        self.centre = centre
        self.size = size

    def isincheck(self, nod: Point):
        # nod - координаты узла
        """Определаяем внешний/внутренний ли узел по
        его координатам и координатам центра сферы и её радиуса
        True, если узел внутренний (за пределами границы (либо на ней), внутри расчётной области)"""
        return (nod - self.centre).square_of_norm() >= self.size ** 2

    def nearbordercheck(self, nod: Point, v: Vector) -> float:
        # nod - координаты узла, v - базисный вектор
        """Определаяем, граничит ли данный узел с границей сферы
         в пределах и направлении заданного вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел НЕ граничит в этом направлении,
        или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению базисного вектора)
        в пределах расстояния между узлами в этом направлении."""
        if (nod - self.centre).square_of_norm() == self.size ** 2:
            return 0
        ans1 = self.isincheck(nod)
        ans2 = self.isincheck(Point(nod.x + v.x * self.lattice.incr[0],
                                    nod.y + v.y * self.lattice.incr[1],
                                    nod.z + v.z * self.lattice.incr[2]))
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0
        delta = nod - self.centre
        a = v.square_of_norm()
        b = (v * delta) * 2
        c = (delta * delta) - self.size ** 2
        det = (b ** 2 - 4 * a * c)
        if det >= 0:
            d1 = (- b + det ** 0.5) / (2 * a)
            d2 = (- b - det ** 0.5) / (2 * a)
            if d1 >= 0 and d2 >= 0:
                return min(d1, d2) * (a ** 0.5)
            elif d1 >= 0 or d2 >= 0:
                return max(d1, d2) * (a ** 0.5)
            else:
                return 0
        else:
            return 0

    def print_normal_and_distance(self, nod: Point):
        # nod - координаты узла
        """Вычисляет вектор нормали к поверхности для сферы, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        delta = self.centre - nod
        length = delta.square_of_norm() ** 0.5
        if length == self.size:  # если узел лежит ровно на сфере
            # возвращаем нулевые вектор нормали и расстояние:
            return Vector(0, 0, 0), 0
        else:
            if length >= self.size:  # внутренние узлы (снаружи сферы)
                # возвращаем вектор нормали и расстояние:
                return Vector(delta.x / length, delta.y / length, delta.z / length), (length - self.size)
            else:  # граничные узлы (внутри сферы)
                # возвращаем вектор нормали и расстояние:
                return Vector(-delta.x / length, -delta.y / length, -delta.z / length), (self.size - length)


class Ellipsoid(Figure):
    def __init__(self, centre=Point(0, 0, 0), size: tuple = (1, 1, 1), lattice: Lattice = None):
        super().__init__(lattice)
        self.centre = centre
        self.size = size

    def isincheck(self, nod: Point):
        # nod - координаты узла
        """Определаяем внешний/внутренний ли узел по
        его координатам и координатам центра эллипсоида и её радиуса
        True, если узел внутренний (за пределами границы (либо на ней), внутри расчётной области)"""
        a = self.size[0]
        b = self.size[1]
        c = self.size[2]
        delta = (nod - self.centre)
        return delta.x ** 2 / a ** 2 + delta.y ** 2 / b ** 2 + delta.z ** 2 / c ** 2 >= 1

    def nearbordercheck(self, nod: Point, v: Vector) -> float:
        # nod - координаты узла, v - базисный вектор
        """Определаяем, граничит ли данный узел с границей эллипсоида
         в пределах и направлении заданного вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел НЕ граничит в этом направлении,
        или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению базисного вектора)
        в пределах расстояния между узлами в этом направлении."""
        delta = nod - self.centre
        a = self.size[0]
        b = self.size[1]
        c = self.size[2]
        if delta.x ** 2 / a ** 2 + delta.y ** 2 / b ** 2 + delta.z ** 2 / c ** 2 == 1:
            return 0
        ans1 = self.isincheck(nod)
        ans2 = self.isincheck(Point(nod.x + v.x * self.lattice.incr[0],
                                    nod.y + v.y * self.lattice.incr[1],
                                    nod.z + v.z * self.lattice.incr[2]))
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0
        a_a = v.square_of_norm()
        b_b = (v * delta) * 2
        c_c = (delta * delta) - self.size ** 2
        det = (b_b ** 2 - 4 * a_a * c_c)
        if det >= 0:
            d1 = (- b_b + det ** 0.5) / (2 * a_a)
            d2 = (- b_b - det ** 0.5) / (2 * a_a)
            if d1 >= 0 and d2 >= 0:
                return min(d1, d2) * (a_a ** 0.5)
            elif d1 >= 0 or d2 >= 0:
                return max(d1, d2) * (a_a ** 0.5)
            else:
                return 0
        else:
            return 0

    def print_normal_and_distance(self, nod: Point):
        # nod - координаты узла
        """Вычисляет вектор нормали к поверхности для эллипсоида, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        delta = self.centre - nod
        length = delta.square_of_norm() ** 0.5
        if length == self.size:  # если узел лежит ровно на эллипсоиде
            # возвращаем нулевые вектор нормали и расстояние:
            return Vector(0, 0, 0), 0
        else:
            if length >= self.size:  # внутренние узлы (снаружи эллипсоида)
                # возвращаем вектор нормали и расстояние:
                return Vector(delta.x / length, delta.y / length, delta.z / length), (length - self.size)
            else:  # граничные узлы (внутри эллипсоида)
                # возвращаем вектор нормали и расстояние:
                return Vector(-delta.x / length, -delta.y / length, -delta.z / length), (self.size - length)


class Parallelepiped(Figure):
    def __init__(self, centre=Point(0, 0, 0), size: tuple = (1, 1, 1), lattice: Lattice = None):
        super().__init__(lattice)
        self.centre = centre
        self.size = size

    def isincheck(self, nod: Point):
        # nod - координаты узла
        """ Определаяем внутренний/внешний ли узел по
        его координатам и координатам центра параллелограмма и размеру его грани
        True, если узел внутренний (за пределами границы (либо на ней), внутри расчётной области)"""
        return (abs(nod.x - self.centre.x) >= self.size[0] / 2 or
                abs(nod.y - self.centre.y) >= self.size[1] / 2 or
                abs(nod.z - self.centre.z) >= self.size[2] / 2)

    def nearbordercheck(self, nod: Point, v: Vector) -> float:
        # nod - координаты узла, v - базисный вектор
        """Определаяем, граничит ли данный узел с границей параллелограмма
         в пределах и направлении заданного вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел НЕ граничит в этом направлении,
        или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению базисного вектора)
        в пределах расстояния между узлами в этом направлении."""
        delta = nod - self.centre
        # размеры параллелограмма:
        a = self.size[0]
        b = self.size[1]
        c = self.size[2]
        if (abs(delta.x) == a / 2 or
            abs(delta.y) == b / 2 or
            abs(delta.z) == c / 2) and (
                abs(delta.x) <= a / 2) and (
                abs(delta.y) <= b / 2) and (
                abs(delta.z) <= c / 2):
            return 0

        ans1 = self.isincheck(nod)
        ans2 = self.isincheck(Point(nod.x + v.x * self.lattice.incr[0],
                                    nod.y + v.y * self.lattice.incr[1],
                                    nod.z + v.z * self.lattice.incr[2]))
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0

        # вычисляем по скольки координатам точка граничит с кубом в пределах узла
        # и расстояние до границы:
        numoftrue = 0
        nod_delta_a = [abs(abs(delta.x) - a / 2),
                       abs(abs(delta.y) - b / 2),
                       abs(abs(delta.z) - c / 2)]
        ans = [False for _ in range(3)]
        for n in range(3):
            if nod_delta_a[n] <= self.lattice.incr[n]:
                numoftrue += 1
                ans[n] = True
        # вычисляем расстояние до параллелограмма для разных вариантов расположения узла:
        xoryorz_dist = 0  # сначала находим расстояние до границы по одной из координат, которая определяет
        # длину всего вектора до границы для конкретного расположения узла относительно параллелограмма
        match numoftrue:
            case 1:
                xoryorz_dist = ans[0] * nod_delta_a[0] + ans[1] * nod_delta_a[1] + ans[2] * nod_delta_a[2]
            case 2:
                if ans1:  # если снаружи параллелограмма
                    xoryorz_dist = max(ans[0] * nod_delta_a[0], ans[1] * nod_delta_a[1], ans[2] * nod_delta_a[2])
                elif ans2:  # если внутри параллелограмма
                    # находим, какие из компонент (по осям) базисных векторов
                    # смотрят от этой точки в сторону границы параллелограмма по этой оси (ближайшей в пределах шага):
                    dist_by_axis = []
                    for n, param in enumerate(delta.__dict__.keys()):
                        chek = ans[n] * (self.lattice.sign_of_num(v.__dict__[param]) ==
                                         self.lattice.sign_of_num(delta.__dict__[param]))
                        if chek:
                            dist_by_axis.append(nod_delta_a[n])
                    xoryorz_dist = dist_by_axis[0] if len(dist_by_axis) == 1 else min(dist_by_axis)
            case 3:
                if ans1:  # если снаружи параллелограмма
                    xoryorz_dist = max(nod_delta_a)
                elif ans2:  # если внутри параллелограмма
                    # находим, какие из компонент (по осям) базисных векторов
                    # смотрят от этой точки в сторону границы параллелограмма по этой оси (ближайшей в пределах шага):
                    dist_by_axis = []
                    for n, param in enumerate(delta.__dict__.keys()):
                        chek = self.lattice.sign_of_num(v.__dict__[param]) == self.lattice.sign_of_num(
                            delta.__dict__[param])
                        if chek:
                            dist_by_axis.append(nod_delta_a[n])
                    xoryorz_dist = dist_by_axis[0] if len(dist_by_axis) == 1 else min(dist_by_axis)
        distance = xoryorz_dist * ((v * v) ** 0.5)
        return distance

    def print_normal_and_distance(self, nod: Point):
        # nod - координаты узла
        """Вычисляет вектор нормали к поверхности для параллелограмма, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        dx = self.centre.x - nod.x
        dy = self.centre.y - nod.y
        dz = self.centre.z - nod.z
        a = self.size[0]
        b = self.size[1]
        c = self.size[2]
        ans1 = abs(dx) == a / 2
        ans2 = abs(dy) == b / 2
        ans3 = abs(dz) == c / 2
        ans4 = abs(dx) <= a / 2
        ans5 = abs(dy) <= b / 2
        ans6 = abs(dz) <= c / 2
        if (ans1 and ans5 and ans6) or (ans2 and ans4 and ans6) or (ans3 and ans4 and ans5):
            # возвращаем вектор нормали и расстояние:
            return Vector(0, 0, 0), 0
        else:
            norm_x, norm_y, norm_z, length = 0, 0, 0, 1
            if self.isincheck(nod):  # внутренние узлы (снаружи параллелограмма)
                # ищем вектор нормали к ближайшей границе (единичный) и расстояние
                # для разных положений точки относительно параллелограмма:
                if ans5 and ans6:
                    length = abs(dx) - a / 2
                    norm_x = dx / abs(dx)
                    norm_y = 0
                    norm_z = 0
                elif ans4 and ans6:
                    length = abs(dy) - b / 2
                    norm_x = 0
                    norm_y = dy / abs(dy)
                    norm_z = 0
                elif ans4 and ans5:
                    length = abs(dz) - c / 2
                    norm_x = 0
                    norm_y = 0
                    norm_z = dz / abs(dz)
                elif ans4 and not ans5 and not ans6:
                    length = ((abs(dy) - b / 2) ** 2 + (abs(dz) - c / 2) ** 2) ** 0.5
                    norm_x = 0
                    norm_y = (dy - self.lattice.sign_of_num(dy) * b / 2) / length
                    norm_z = (dz - self.lattice.sign_of_num(dz) * c / 2) / length
                elif ans5 and not ans4 and not ans6:
                    length = ((abs(dx) - a / 2) ** 2 + (abs(dz) - c / 2) ** 2) ** 0.5
                    norm_x = (dx - self.lattice.sign_of_num(dx) * a / 2) / length
                    norm_y = 0
                    norm_z = (dz - self.lattice.sign_of_num(dz) * c / 2) / length
                elif ans6 and not ans4 and not ans5:
                    length = ((abs(dx) - a / 2) ** 2 + (abs(dy) - b / 2) ** 2) ** 0.5
                    norm_x = (dx - self.lattice.sign_of_num(dx) * a / 2) / length
                    norm_y = (dy - self.lattice.sign_of_num(dy) * b / 2) / length
                    norm_z = 0
                elif not ans4 and not ans5 and not ans6:
                    length = ((abs(dx) - a / 2) ** 2 + (abs(dy) - b / 2) ** 2 + (abs(dz) - c / 2) ** 2) ** 0.5
                    norm_x = (dx - self.lattice.sign_of_num(dx) * a / 2) / length
                    norm_y = (dy - self.lattice.sign_of_num(dy) * b / 2) / length
                    norm_z = (dz - self.lattice.sign_of_num(dz) * c / 2) / length
                # возвращаем вектор нормали и расстояние:
                return Vector(norm_x, norm_y, norm_z), length
            else:  # граничные узлы (внутри параллелограмма)
                dx *= -1
                dy *= -1
                dz *= -1
                an1 = (a / 2 - abs(dx)) <= (b / 2 - abs(dy))
                an2 = (a / 2 - abs(dx)) <= (c / 2 - abs(dz))
                an3 = (b / 2 - abs(dy)) <= (a / 2 - abs(dx))
                an4 = (b / 2 - abs(dy)) <= (c / 2 - abs(dz))
                an5 = (c / 2 - abs(dz)) <= (a / 2 - abs(dx))
                an6 = (c / 2 - abs(dz)) <= (b / 2 - abs(dy))
                if an1 and an2:
                    length = a / 2 - abs(dx)
                    norm_x = dx / abs(dx)
                    norm_y = 0
                    norm_z = 0
                elif an3 and an4:
                    length = b / 2 - abs(dy)
                    norm_x = 0
                    norm_y = dy / abs(dy)
                    norm_z = 0
                elif an5 and an6:
                    length = c / 2 - abs(dz)
                    norm_x = 0
                    norm_y = 0
                    norm_z = dz / abs(dz)
                # возвращаем вектор нормали и расстояние:
                return Vector(norm_x, norm_y, norm_z), length


good_lattice = Lattice()
good_lattice.create_outfile()
