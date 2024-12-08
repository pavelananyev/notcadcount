from decimal import *

getcontext().prec = 30


class Point:
    def __init__(self, x, y: int | float | Decimal = 0, z: int | float | Decimal = 0):
        if isinstance(x, (int, float, Decimal)) and isinstance(y, (int, float, Decimal)) and isinstance(z, (
                int, float, Decimal)):
            self.x = x
            self.y = y
            self.z = z
        elif isinstance(x, (tuple, list)):  # если передаём точку готовым кортежем или списком координат
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
    def __init__(self, x: int | float | Decimal = 0, y: int | float | Decimal = 0, z: int | float | Decimal = 0):
        if isinstance(x, (int, float, Decimal)) and isinstance(y, (int, float, Decimal)) and isinstance(z, (
                int, float, Decimal)):
            self.x = x
            self.y = y
            self.z = z
        else:
            raise ValueError("Компоненты вектора должны быть числами!")

    def __add__(self, other):  # при сложении с точкой возвращает новую точку
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z) if isinstance(other, Point) else Vector(x, y, z)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Vector(x, y, z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif isinstance(other, (int, float, Decimal)):
            return Vector(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Неподходящий тип данных для умножения на вектор")

    def __truediv__(self, other):
        if isinstance(other, (int, float, Decimal)):
            return Vector(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Вектор можно поделить только на число")

    def __pow__(self, other):
        if isinstance(other, int) and other > 0:
            out = 1
            for _ in range(other):
                out = self.__mul__(out)
            return out
        else:
            raise ValueError("Вектора можно возводить только в целую положительную степень")

    def square_of_norm(self):
        return self * self


class Lattice:
    def __init__(self, ):
        self.fgr = None
        self.nodes_to_write = None
        self.z_min = None
        self.y_min = None
        self.x_min = None
        self.Nz = None
        self.Ny = None
        self.Nx = None
        self.num_of_nodes = None
        self.border_type = None
        self.lattice_num = None
        self.nums = None

    def lattice_generator(self, lattices_):
        """Вычисление количества и генерация узлов решётки в соответствии с параметрами,
        с отсевом и сортировкой по разным категориям"""
        print(f'Вложенность решётки: {nested_lattice_quantity}')
        # xyz_borders = [0, 0, 0, 0, 0, 0] # прямоугольные границы данного уровня решётки
        if nested_lattice_quantity == 0:
            self.num_of_nodes = []
            for n in range(6):  # вычисляем и складываем количество узлов
                # по разные стороны центра фигуры границы, для каждой оси.
                # На данный момент узел решётки совпадает с центром фигуры границы
                # и откладывается от него до границ области вычислений + 2 узла на внешние границы.
                centre = (figure.centre.x, figure.centre.y, figure.centre.z)
                self.num_of_nodes.append(int(abs((minmax_coord[n] - centre[n // 2]) / incr[n // 2])))
            self.Nx = self.num_of_nodes[0] + self.num_of_nodes[1] + 1 + 2  # + 2 это на внешние границы добавили
            self.Ny = self.num_of_nodes[2] + self.num_of_nodes[3] + 1 + 2  # + 2 это на внешние границы добавили
            self.Nz = self.num_of_nodes[4] + self.num_of_nodes[5] + 1 + 2  # + 2 это на внешние границы добавили

            # Координаты левого нижнего узла решётки - граничного
            # внешнего узла за пределами расчётной области (координаты узла по каждому направлению Ox, Oy, Oz)
            # координаты внутренних узлов на 1 шаг ближе к центру по соответствующим координатам и находятся
            # скраю расчетной области
            self.x_min = figure.centre.x - (self.num_of_nodes[0] + 1) * incr[0]
            self.y_min = figure.centre.y - (self.num_of_nodes[2] + 1) * incr[1]
            self.z_min = figure.centre.z - (self.num_of_nodes[4] + 1) * incr[2]

            # В этом блоке отсекаются все внешние узлы и сохраняются внешние граничные узлы внутри поверхности,
            # в виде списка кортежей их координат. И распределяются по этим группам.
            self.nums = [0, 0, self.Ny * self.Nz, self.Ny * self.Nz, (self.Nx - 2) * self.Nz,
                         (self.Nx - 2) * self.Nz, (self.Nx - 2) * (self.Ny - 2), (self.Nx - 2) * (self.Ny - 2)]
            self.nodes_to_write = [[] for _ in range(8)]
            inside_out_nds = 0
            for k in range(1, self.Nz - 1):  # генерируем узлы в расчётной области, сюда входит и то, что внутри
                # поверхности. Оно в процессе отсеивается, или идёт в границу расчётной области у поверхности.
                # убираем из циклов крайние значения - узлы внешних границ решётки,т.к. они отдельно ниже генерируются
                z = self.z_min + k * incr[2]
                for j in range(1, self.Ny - 1):
                    y = self.y_min + j * incr[1]
                    for i in range(1, self.Nx - 1):
                        x = self.x_min + i * incr[0]
                        if figure.isincheck(Point(x, y, z)):
                            self.nums[0] += 1
                            self.nodes_to_write[0].append((i, j, k))
                        elif figure.isbordercheck(Point(x, y, z)):
                            self.nums[1] += 1
                            self.nodes_to_write[1].append((i, j, k))
                        else:
                            inside_out_nds += 1
            print('Не граничных узлов внутри поверхности :', inside_out_nds)
            sum1 = self.Nx * self.Ny * self.Nz
            sum2 = sum(self.nums) + inside_out_nds
            print(
                f'_______________________________ПРОВЕРКА \"ЦЕЛОСТНОСТИ\" сетка №{self.lattice_num}'
                f'______________________________\n'
                f'узлов посчитано: {self.Nx} * {self.Ny} * {self.Nz} = {sum1}\n'
                f'узлов создано: {self.nums[0]} + {inside_out_nds} + {self.nums[1]} + {self.nums[2]} + {self.nums[3]}'
                f' + {self.nums[4]} + {self.nums[5]} + {self.nums[6]} + {self.nums[7]} = {sum2}\n'
                f'............................................................................................')
        else:
            pass
        if self.lattice_num == 0:
            # генерируем узлы границ расчётных областей (внешние узлы на границе с расчётной областью)
            for k in range(self.Nz):
                for j in range(self.Ny):
                    self.nodes_to_write[2].append((0, j, k))
                    self.nodes_to_write[3].append((self.Nx - 1, j, k))
            for k in range(self.Nz):
                for i in range(1, self.Nx - 1):
                    self.nodes_to_write[4].append((i, 0, k))
                    self.nodes_to_write[5].append((i, self.Ny - 1, k))
            for j in range(1, self.Ny - 1):
                for i in range(1, self.Nx - 1):
                    self.nodes_to_write[6].append((i, j, 0))
                    self.nodes_to_write[7].append((i, j, self.Nz - 1))

        if self.lattice_num < nested_lattice_quantity:
            lattices_.append(lattices_[-1])

    def create_outfile(self, j: int):
        """Вызываем все функции, делаем вычисления и записываем в файл"""
        try:
            with open(path_out[:-4] + f'_{j}.txt', "w") as file:
                file.write(
                    f'{self.x_min};{self.y_min};{self.z_min}\n')  # Координаты левого нижнего узла решётки - граничного
                # внешнего узла за пределами расчётной области (координаты узла по каждому направлению Ox, Oy, Oz)
                # координаты внутренних узлов на 1 шаг ближе к центру по соответствующим координатам и находятся
                # скраю расчетной области

                file.write(f'{incr[0]};{incr[1]};{incr[2]}\n')  # шаг решётки по каждому
                # направлению
                file.write(f'{self.Nx};{self.Ny};{self.Nz}\n')  # общее количество узлов по каждой оси

                # Количество внутренних узлов, и далее через ; количество узлов
                # для каждой из границ расчётной области. Сначала идут граничные узлы вдоль поверхности (они являются
                # внешними, лежат уже внутри поверхности), потом внешние границы (тоже внешние узлы).

                # Количество внутренних узлов записываем:
                file.write(f'{self.nums[0]}')
                # и количество граничных узлов записываем:
                for i in range(1, 8):
                    file.write(f';{self.nums[i]}')
                file.write('\n')

                # Далее всё повторяется для каждого последующего вектора:
                #     id; i; j; k;
                #     26 * расстояний (от узла до границы, в пределах одного узла) по 26 заданным направлениям;
                #     вектор нормали к ближайшей границе (единичный);
                #     расстояние до ближайшей границы (по вектору нормали будет).
                #
                #     Идентификатор узла определяется id = i + (j + k * Ny) * Nx,
                #     сначала список узлов внутренних, потом по очереди всех границ
                counter = 0
                for n in range(8):
                    # сначала для внутренних узлов считаем и записываем (n=0), затем для граничных узлов
                    # поверхности (n=1), затем для границ расчётной области (n > 1):
                    for i, j, k in self.nodes_to_write[n]:
                        id_ = i + (j + k * self.Ny) * self.Nx
                        file.write(f'{id_};{i};{j};{k}')
                        x = self.x_min + i * incr[0]
                        y = self.y_min + j * incr[1]
                        z = self.z_min + k * incr[2]
                        # 26 * расстояний по 26 заданным направлениям:
                        for vector in basis:
                            file.write(f';{figure.nearbordercheck(Point(x, y, z), vector)}')
                        # вектор нормали к ближайшей границе (единичный)
                        # и расстояние до ближайшей границы (по вектору нормали):
                        normal, distance = figure.print_normal_and_distance(Point(x, y, z))
                        file.write(f';{normal.x};{normal.y};{normal.z};{distance}')
                        file.write(f'\n')
                        counter += 1
                print(f'Записалось узлов в выходном файле:  {counter + 4} - 4 = {counter}\n'
                      f'Должно совпасть с суммой узлов без \"не граничных внутри поверхности\": {sum(self.nums)}\n'
                      f'____________________________________________________________________________________________')
        except Exception as error:
            print(f'Ошибка при создании выходного файла:\n{error}')


class Figure:
    def __init__(self):
        pass

    def isincheck(self, *args):
        raise NotImplementedError("В дочернем классе должен быть"
                                  "переопределён метод isincheck()")

    def isbordercheck(self, node: Point):
        # node - координаты внешнего узла (вне расчётной области, внутри поверхности)
        """Определаяем, граничит ли внешний узел с границей поверхности
        (а значит с внутренним узлом) хотя бы по одному базисному вектору в пределах шага решётки;
        True, если узел граничит с поверхностью"""
        for v in basis:
            ans = self.isincheck(Point(node.x + v.x * incr[0],
                                       node.y + v.y * incr[1],
                                       node.z + v.z * incr[2]))
            if ans:
                return True
        return False


class Sphere(Figure):
    def __init__(self, centre=Point(0, 0, 0), size=None):
        super().__init__()
        self.centre = centre
        self.size = size

    def isincheck(self, node: Point):
        # node - координаты узла
        """Определаяем внешний/внутренний ли узел по
        его координатам и координатам центра сферы и её радиуса
        True, если узел внутренний (за пределами границы (либо на ней), внутри расчётной области)"""
        return (node - self.centre).square_of_norm() >= self.size ** 2

    def nearbordercheck(self, node: Point, v: Vector) -> float:
        # node - координаты узла, v - базисный вектор
        """Определаяем, граничит ли данный узел с границей сферы
         в пределах и направлении заданного БАЗИСНОГО вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел НЕ граничит в этом направлении,
        или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению базисного вектора)
        в пределах расстояния между узлами в этом направлении."""
        if (node - self.centre).square_of_norm() == self.size ** 2:
            return 0
        ans1 = self.isincheck(node)
        ans2 = self.isincheck(Point(node.x + v.x * incr[0],
                                    node.y + v.y * incr[1],
                                    node.z + v.z * incr[2]))
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0
        delta = node - self.centre
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

    def print_normal_and_distance(self, node: Point):
        # node - координаты узла
        """Вычисляет вектор нормали от точки к поверхности для сферы, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        delta = self.centre - node
        length = delta.square_of_norm() ** 0.5
        if length == self.size:  # если узел лежит ровно на сфере
            # возвращаем нулевые вектор нормали и расстояние:
            return Vector(0, 0, 0), 0
        else:
            if length > self.size:  # внутренние узлы (снаружи сферы)
                # возвращаем вектор нормали и расстояние:
                return Vector(delta.x / length, delta.y / length, delta.z / length), (length - self.size)
            else:  # граничные узлы (внутри сферы)
                # возвращаем вектор нормали и расстояние:
                return Vector(-delta.x / length, -delta.y / length, -delta.z / length), (self.size - length)


class Ellipsoid(Figure):
    def __init__(self, centre=Point(0, 0, 0), size: tuple = (1, 1, 1)):
        super().__init__()
        self.centre = centre
        self.size = size

    def isincheck(self, node: Point):
        # node - координаты узла
        """Определаяем внешний/внутренний ли узел по
        его координатам и координатам центра эллипсоида и её радиуса
        True, если узел внутренний (за пределами границы (либо на ней), внутри расчётной области)"""
        delta = (node - self.centre)
        return delta.x ** 2 / self.size[0] ** 2 + delta.y ** 2 / self.size[1] ** 2 + delta.z ** 2 / self.size[
            2] ** 2 >= 1

    @staticmethod
    def intersection_with_line_calculate(k1: Decimal, k2: Decimal, a: Decimal, b: Decimal, c: Decimal, p1: Point,
                                         p0: Point) -> tuple[Vector | None, Decimal | None]:
        """ Функция вычисляет ближайшую к точке p1 точку пересечения прямой и эллипсоида, если она есть, и возвращает
        вектор от p1 до этой точки и расстояние от неё до p1 (длину вектора), либо None.
        Коэффициенты k1 и k2 - из выражений преобразования координат в системе линейных уравнений
        для пересечения эллипсоида и прямой:
        (x - x1) = k1 * (y - y1)
        (z - z1) = k2 * (y - y1)
        (x - x0)**2 / a**2 + (y - y0)**2 / b**2 + (z - z0)**2 / c**2 - 1 = 0
        Расчёты выполнены через координату y с последующей подстановкой в x и z.

        a, b, c - коэффициенты из уравнения эллипсоида
        p1 = Point(x1, y1, z1) - известная точка не на эллипсоиде, через которую проходит пересекающая эллипсоид прямая
        p0 = Point(x0, y0, z0) - точка центра эллипсоида

        ВАЖНО: при использовании функции, в зависимости от начальных условий, подставляемые в неё координаты точек
        p0 и p1 и всех коэффициентов могут быть "перемешаны", т.е. x, y и z, а также аналогично a, b, и c фактически
        будут поменяны местами. В силу симметричности задачи относительно осей координат, такое использование удобно и
        сокращает код. В основном коде после обращения к этой функции тогда идёт обратная подмена координат, чтобы
        вернуть верное соответствие значений осям.
        Требуется такая "подмена", когда нужно избежать деления на ноль и выбрать ненулевую координату у базисного
        вектора для дальнейших расчётов по ней."""
        b1 = Decimal(p1.x) - k1 * Decimal(p1.y)
        b2 = Decimal(p1.z) - k2 * Decimal(p1.y)
        s1 = b1 - Decimal(p0.x)
        s2 = b2 - Decimal(p0.z)
        v = (a * b * c) ** 2
        a_a = (a * c) ** 2 + (b * c * k1) ** 2 + (a * b * k2) ** 2
        b_b = 2 * (((b * c) ** 2) * k1 * s1 + ((a * b) ** 2) * k2 * s2 - ((a * c) ** 2) * Decimal(p0.y))
        c_c = (a * c * Decimal(p0.y)) ** 2 + (b * c * s1) ** 2 + (a * b * s2) ** 2 - v
        det = b_b ** 2 - 4 * a_a * c_c
        print(
            f'b1 = {b1}\nb2 = {b2}\ns1 = {s1}\ns2 = {s2}\nv = {v}\nA = {a_a}\nB = {b_b}\nC = {c_c}\n'
            f'D = {det}\nNode = {p1.x, p1.y, p1.z}')
        if det < Decimal('0'):
            return None, None
        else:
            y_1 = (- b_b + det ** Decimal(0.5)) / (2 * a_a)
            x_1 = k1 * y_1 + b1
            z_1 = k2 * y_1 + b2
            norm1 = Point(x_1, y_1, z_1) - Point(Decimal(p1.x), Decimal(p1.y), Decimal(p1.z))
            square_dist1 = norm1.square_of_norm()
            print(x_1, y_1, z_1, norm1, square_dist1)
            y_2 = (- b_b - det ** Decimal(0.5)) / (2 * a_a)
            x_2 = k1 * y_2 + b1
            z_2 = k2 * y_2 + b2
            norm2 = Point(x_2, y_2, z_2) - Point(Decimal(p1.x), Decimal(p1.y), Decimal(p1.z))
            square_dist2 = norm2.square_of_norm()
            print(x_2, y_2, z_2, norm2, square_dist2)
            if square_dist1 < square_dist2:
                print(f'norm = {norm1.x, norm1.y, norm1.z}, dist = {square_dist1 ** Decimal(0.5)}')
                return norm1, square_dist1 ** Decimal(0.5)
            else:
                print(f'norm = {norm2.x, norm2.y, norm2.z}, dist = {square_dist2 ** Decimal(0.5)}')
                return norm2, square_dist2 ** Decimal(0.5)

    def nearbordercheck(self, node: Point, v: Vector) -> float:
        # node - координаты узла, v - базисный вектор
        """Определаяем, граничит ли данный узел с границей эллипсоида
         в пределах и направлении заданного БАЗИСНОГО вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел НЕ граничит в этом направлении,
        или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению базисного вектора)
        в пределах расстояния между узлами в этом направлении."""
        delta = node - self.centre
        a = Decimal(self.size[0])
        b = Decimal(self.size[1])
        c = Decimal(self.size[2])
        if (Decimal(delta.x) ** 2 / a ** 2 + Decimal(delta.y) ** 2 / b ** 2 + Decimal(
                delta.z) ** 2 / c ** 2) == Decimal('1'):
            return 0
        p = Point(node.x + v.x * incr[0], node.y + v.y * incr[1],
                  node.z + v.z * incr[2])
        ans1 = self.isincheck(node)
        ans2 = self.isincheck(p)
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0
        if v.x != 0:
            x1, y1, z1 = node.y, node.x, node.z
            x2, y2, z2 = p.y, p.x, p.z
            a, b, c = b, a, c
            p0 = Point(self.centre.y, self.centre.x, self.centre.z)
        elif v.y != 0:
            x1, y1, z1 = node.x, node.y, node.z
            x2, y2, z2 = p.x, p.y, p.z
            p0 = self.centre
        else:
            x1, y1, z1 = node.x, node.z, node.y
            x2, y2, z2 = p.x, p.z, p.y
            a, b, c = a, c, b
            p0 = Point(self.centre.x, self.centre.z, self.centre.y)
        k1 = (Decimal(x2) - Decimal(x1)) / (Decimal(y2) - Decimal(y1))
        k2 = (Decimal(z2) - Decimal(z1)) / (Decimal(y2) - Decimal(y1))
        distance = self.intersection_with_line_calculate(k1, k2, a, b, c, Point(x1, y1, z1), p0)[1]
        if distance is None:
            return 0
        else:
            return float(distance)

    def print_normal_and_distance(self, node: Point):
        # node - координаты узла
        """Вычисляет вектор нормали от точки к поверхности для эллипсоида, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        # print("print_normal_and_distance")
        # print((node.x, node.y, node.z))
        delta = self.centre - node
        a = Decimal(self.size[0])
        b = Decimal(self.size[1])
        c = Decimal(self.size[2])
        if (Decimal(delta.x) ** 2 / a ** 2 + Decimal(delta.y) ** 2 / b ** 2 + Decimal(
                delta.z) ** 2 / c ** 2) == Decimal('1'):
            # если узел лежит ровно на эллипсоиде - возвращаем нулевые вектор нормали и расстояние:
            return Vector(0, 0, 0), 0
        else:
            k1 = (a / b) ** 2
            k2 = (c / b) ** 2
            normal, distance = self.intersection_with_line_calculate(k1, k2, a, b, c, node, self.centre)
            if normal is None:
                return Vector(999, 999, 999), 999
                # raise ArithmeticError("Не смог вычислить нормальный вектор от точки")
            normal = Vector(float(normal.x / distance), float(normal.y / distance), float(normal.z / distance))
            distance = float(distance)
            return normal, distance


class Parallelepiped(Figure):
    def __init__(self, centre=Point(0, 0, 0), size: tuple = (1, 1, 1)):
        super().__init__()
        self.centre = centre
        self.size = size

    def isincheck(self, node: Point):
        # node - координаты узла
        """ Определаяем внутренний/внешний ли узел по
        его координатам и координатам центра параллелограмма и размеру его грани
        True, если узел внутренний (за пределами границы (либо на ней), внутри расчётной области)"""
        return (abs(node.x - self.centre.x) >= self.size[0] / 2 or
                abs(node.y - self.centre.y) >= self.size[1] / 2 or
                abs(node.z - self.centre.z) >= self.size[2] / 2)

    def nearbordercheck(self, node: Point, v: Vector) -> float:
        # node - координаты узла, v - базисный вектор
        """Определаяем, граничит ли данный узел с границей параллелограмма
         в пределах и направлении заданного БАЗИСНОГО вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел НЕ граничит в этом направлении,
        или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению базисного вектора)
        в пределах расстояния между узлами в этом направлении."""
        delta = node - self.centre
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

        ans1 = self.isincheck(node)
        ans2 = self.isincheck(Point(node.x + v.x * incr[0],
                                    node.y + v.y * incr[1],
                                    node.z + v.z * incr[2]))
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            return 0

        # вычисляем по скольки координатам точка граничит с кубом в пределах узла
        # и расстояние до границы:
        numoftrue = 0
        node_delta_a = [abs(abs(delta.x) - a / 2),
                        abs(abs(delta.y) - b / 2),
                        abs(abs(delta.z) - c / 2)]
        ans = [False for _ in range(3)]
        for n in range(3):
            if node_delta_a[n] <= incr[n]:
                numoftrue += 1
                ans[n] = True
        # вычисляем расстояние до параллелограмма для разных вариантов расположения узла:
        xoryorz_dist = 0  # сначала находим расстояние до границы по одной из координат, которая определяет
        # длину всего вектора до границы для конкретного расположения узла относительно параллелограмма
        match numoftrue:
            case 1:
                xoryorz_dist = ans[0] * node_delta_a[0] + ans[1] * node_delta_a[1] + ans[2] * node_delta_a[2]
            case 2:
                if ans1:  # если снаружи параллелограмма
                    xoryorz_dist = max(ans[0] * node_delta_a[0], ans[1] * node_delta_a[1], ans[2] * node_delta_a[2])
                elif ans2:  # если внутри параллелограмма
                    # находим, какие из компонент (по осям) базисных векторов
                    # смотрят от этой точки в сторону границы параллелограмма по этой оси (ближайшей в пределах шага):
                    dist_by_axis = []
                    for n, param in enumerate(delta.__dict__.keys()):
                        check = ans[n] * (sign_of_num(v.__dict__[param]) == sign_of_num(delta.__dict__[param]))
                        if check:
                            dist_by_axis.append(node_delta_a[n])
                    xoryorz_dist = dist_by_axis[0] if len(dist_by_axis) == 1 else min(dist_by_axis)
            case 3:
                if ans1:  # если снаружи параллелограмма
                    xoryorz_dist = max(node_delta_a)
                elif ans2:  # если внутри параллелограмма
                    # находим, какие из компонент (по осям) базисных векторов
                    # смотрят от этой точки в сторону границы параллелограмма по этой оси (ближайшей в пределах шага):
                    dist_by_axis = []
                    for n, param in enumerate(delta.__dict__.keys()):
                        check = sign_of_num(v.__dict__[param]) == sign_of_num(delta.__dict__[param])
                        if check:
                            dist_by_axis.append(node_delta_a[n])
                    xoryorz_dist = dist_by_axis[0] if len(dist_by_axis) == 1 else min(dist_by_axis)
        distance = xoryorz_dist * ((v * v) ** 0.5)
        return distance

    def print_normal_and_distance(self, node: Point):
        # node - координаты узла
        """Вычисляет вектор нормали от точки к поверхности для параллелограмма, затем возвращает его координаты и длину.
        Если точка ровно на поверхности - возвращает нули"""
        dx = self.centre.x - node.x
        dy = self.centre.y - node.y
        dz = self.centre.z - node.z
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
            if self.isincheck(node):  # внутренние узлы (снаружи параллелограмма)
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
                    norm_y = (dy - sign_of_num(dy) * b / 2) / length
                    norm_z = (dz - sign_of_num(dz) * c / 2) / length
                elif ans5 and not ans4 and not ans6:
                    length = ((abs(dx) - a / 2) ** 2 + (abs(dz) - c / 2) ** 2) ** 0.5
                    norm_x = (dx - sign_of_num(dx) * a / 2) / length
                    norm_y = 0
                    norm_z = (dz - sign_of_num(dz) * c / 2) / length
                elif ans6 and not ans4 and not ans5:
                    length = ((abs(dx) - a / 2) ** 2 + (abs(dy) - b / 2) ** 2) ** 0.5
                    norm_x = (dx - sign_of_num(dx) * a / 2) / length
                    norm_y = (dy - sign_of_num(dy) * b / 2) / length
                    norm_z = 0
                elif not ans4 and not ans5 and not ans6:
                    length = ((abs(dx) - a / 2) ** 2 + (abs(dy) - b / 2) ** 2 + (abs(dz) - c / 2) ** 2) ** 0.5
                    norm_x = (dx - sign_of_num(dx) * a / 2) / length
                    norm_y = (dy - sign_of_num(dy) * b / 2) / length
                    norm_z = (dz - sign_of_num(dz) * c / 2) / length
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


def sign_of_num(x: float):
    return 0 if abs(x) == 0 else int(x / abs(x))


def get_input():
    """Достаём все параметры из входного файла"""
    try:
        with open(path_in, encoding='utf-8') as file:
            lines = file.readlines()
            minmax_crd = tuple(map(float, lines[0].split(';')))  # крайние
            # координаты (мин, макс) области вычислений по осям координат
            print('Крайние координаты (мин, макс) области вычислений: ', minmax_crd)
            incrmnt = tuple(map(float, lines[1].split(';')))  # шаги решётки по осям координат
            print('Шаги решётки по осям координат: ', incrmnt)
            border_type = int(lines[2].strip())  # тип фигуры для задания границы
            # вытаскиваем параметры для каждого типа фигуры
            t = tuple(map(float, lines[3].split(';')))
            # вытаскиваем параметры для вложенных решёток
            nstd_lttc_qntt, nstd_lttc_wdth = map(int, lines[4].split(';'))
            match border_type:
                case 0:  # произвольная граница с заданием по точкам/аналитически
                    pass
                case 1:  # сфера
                    figure_centre, figure_size = t[:3], t[-1] / 2  # координаты центра сферы
                    # и радиус сферы
                    print(f'СФЕРА с центром {figure_centre} и радиусом {figure_size}')
                    fgr = Sphere(Point(figure_centre), figure_size)

                case 2:  # эллипсоид
                    figure_centre, figure_size = t[:3], tuple(
                        n / 2 for n in t[3::])  # координаты центра эллипсоида
                    # и полуосей a, b, c
                    print(
                        f'ЭЛЛИПСОИД с центром {figure_centre} и полуосями {figure_size[0]}, {figure_size[1]},'
                        f' {figure_size[2]}')
                    fgr = Ellipsoid(Point(figure_centre), figure_size)
                case 3:  # куб
                    figure_centre, figure_size = t[:3], t[-1]  # координаты центра куба
                    # и длина ребра куба
                    print(f'КУБ с центром {figure_centre} и ребром {figure_size}')
                    fgr = Parallelepiped(Point(figure_centre), (figure_size, figure_size, figure_size))
                case 4:  # параллелепипед
                    figure_centre, figure_size = t[:3], tuple(t[3::])  # координаты центра параллелепипеда
                    # и длины рёбер a, b, c
                    print(
                        f'ПАРАЛЛЕЛЕПИПЕД с центром {figure_centre} и рёбрами {figure_size[0]}, {figure_size[1]},'
                        f' {figure_size[2]}')
                    fgr = Parallelepiped(Point(figure_centre), figure_size)
    except Exception as error:
        print(f'Неверный формат входного файла:\n{error}')
    return fgr, minmax_crd, incrmnt, nstd_lttc_qntt, nstd_lttc_wdth


def lattice_compiling():
    """Функция запускает всё необходимое для обработки данных и генерации выходного файла"""
    lattices = [Lattice()]
    for i in range(nested_lattice_quantity + 1):
        lattices[i].lattice_num = i
        lattices[i].lattice_generator(lattices)
        lattices[i].create_outfile(i)
        # lattice = Lattice()
        # lattice.lattice_num = i
        # lattice.lattice_generator()
        # lattice.create_outfile(i)


basis = tuple(Vector(v[0], v[1], v[2]) for v in
              ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, 1, 0),
               (1, -1, 0),
               (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
               (0, -1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1),
               (-1, -1, -1)))
path_in = 'input.txt'
path_out = 'output/output.txt'
figure, minmax_coord, incr, nested_lattice_quantity, nested_lattice_width = get_input()
lattice_compiling()
