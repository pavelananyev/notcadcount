class LatticeCreator:
    BASIS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, 1, 0), (1, -1, 0),
             (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
             (0, -1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1),
             (-1, -1, -1))

    def __init__(self, path_in='input.txt', path_out='output.txt'):
        self.path_in = path_in
        self.path_out = path_out

    def get_input(self):
        """достаём все параметры из входного файла"""
        try:
            with open(self.path_in, encoding='utf-8') as file:
                lines = file.readlines()
                self.minmaxcoord = tuple(map(float, lines[0].split(';')))  # крайние
                # координаты (мин, макс) области вычислений по осям координат
                print('крайние координаты (мин, макс) области вычислений: ', self.minmaxcoord)
                self.incr = tuple(map(float, lines[1].split(';')))  # шаги решётки по осям координат
                print('шаги решётки по осям координат: ', self.incr)
                self.border_type = int(lines[2].strip())  # тип фигуры для задания границы

                self.figure_centre = [0, 0, 0]
                match self.border_type:  # вытаскиваем параметры для каждого типа фигуры
                    case 0:  # произвольная граница с заданием по точкам/аналитически
                        pass
                    case 1:  # сфера
                        t = tuple(map(float, lines[3].split(';')))
                        self.sphere_centre, self.sphere_r = t[:-1], t[-1] / 2  # координаты центра сферы
                        # и радиуса (половина диаметра)
                        self.figure_centre = self.sphere_centre
                        print('сфера: ', self.sphere_centre, self.sphere_r)
                    case 2:  # эллипсоид
                        pass
                    case 3:  # куб
                        t = tuple(map(float, lines[3].split(';')))
                        self.cube_centre, self.cube_size = t[:-1], t[-1]  # координаты центра куба
                        # и длины ребра
                        self.figure_centre = self.cube_centre
                        print('куб: ', self.cube_centre, self.cube_size)
                    case 4:  # параллелепипед
                        pass
        except:
            print('Неверный формат входного файла')

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
    def isincheck_sphere(x, y, z, cx, cy, cz, r):
        """ определаяем внешний/внутренний ли узел по
        его координатам и координатам центра сферы и её радиуса
        True, если узел внутренний (за пределами границы, внутри расчётной области)"""
        return ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5 >= r

    def isbordercheck_sphere(self, x, y, z, cx, cy, cz, r):
        """Определаяем, граничит ли внешний узел с границей сферы (а значит с внутренним узлом)
        хотя бы по одной своей координате; определяем по его координатам и координатам
        центра сферы и её радиуса
        True, если узел граничит со сферой"""
        ans = []
        answer = False
        for vector in self.BASIS:
            ans.append(self.isincheck_sphere(x + self.incr[0] * vector[0], y + self.incr[1] * vector[1],
                                             z + self.incr[2] * vector[2], cx, cy, cz, r))
            answer = answer or ans[-1]
        return answer

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

    def nearbordercheck_sphere(self, o: tuple, u: tuple) -> float:
        # o - координаты узла, u - базисный вектор
        """Определаяем, граничит ли данный узел с границей сферы
         в пределах и направлении заданного вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел не граничит в этом направлении, или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению вектора)
        в пределах расстояния между узлами в этом направлении."""
        sph = self.sphere_centre  # координаты центра сферы
        r = self.sphere_r  # радиус сферы
        if ((o[0] - sph[0]) ** 2 + (o[1] - sph[1]) ** 2 + (o[2] - sph[2]) ** 2) ** 0.5 == r:
            return 0

        x1 = o[0] + u[0] * self.incr[0]
        y1 = o[1] + u[1] * self.incr[1]
        z1 = o[2] + u[2] * self.incr[2]

        ans1 = self.isincheck_sphere(o[0], o[1], o[2], sph[0], sph[1], sph[2], r)
        ans2 = self.isincheck_sphere(x1, y1, z1, sph[0], sph[1], sph[2], r)
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            # if self.m < 1000: # для отлова ошибок принты
            #     self.m += 1
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'*ans1*ans2, end='')
            # print(f'm={self.m}', ' ' * (4 - len(str(self.m))), f'o=', *o, ' ' * (48 - len(str(o))),
            #       f' u={u} {ans1} {ans2}', ' ' * (18 - len(str(u))), f' x={x1} {y1} {z1}')
            return 0

        o_sph = self.vect_sum(o, self.vect_num_mult(-1, sph))
        a = self.vect_scalar_mult(u, u)
        b = 2 * self.vect_scalar_mult(u, o_sph)
        c = self.vect_scalar_mult(o_sph, o_sph) - r ** 2
        det = (b ** 2 - 4 * a * c)
        if det >= 0:
            d1 = (- b + det ** 0.5) / (2 * a)
            d2 = (- b - det ** 0.5) / (2 * a)
            # if self.m < 1000: # для отлова ошибок принты
            #     self.m += 1
            #     print(d1, f'd1 = {d1 * (a ** 0.5)}')
            #     print(d2, f'd2 = {d2 * (a ** 0.5)}')
            #     print(f'm={self.m}', ' ' * (4 - len(str(self.m))), f'o=', *o, ' ' * (48 - len(str(o))),
            #           f' u={u} {ans1} {ans2}', ' ' * (18 - len(str(u))), f' x={x1} {y1} {z1}')
            if d1 >= 0 and d2 >= 0:
                return min(d1 * (a ** 0.5), d2 * (a ** 0.5))
            elif d1 >= 0 or d2 >= 0:
                return max(d1 * (a ** 0.5), d2 * (a ** 0.5))
            else:
                return 0
        else:
            return 0

    @staticmethod
    def isincheck_cube(x, y, z, cx, cy, cz, a):
        """ определаяем внутренний/внешний ли узел по
        его координатам и координатам центра куба и размеру его грани
        True, если узел внутренний (за пределами границы, внутри расчётной области)"""
        return abs(x - cx) >= a / 2 or abs(y - cy) >= a / 2 or abs(z - cz) >= a / 2

    # def isbordercheck_cube(self, x, y, z, cx, cy, cz, a):
    #     '''Определаяем, граничит ли внешний узел с границей куба (внутренним узлом)
    #     хотя бы по одной своей координате;
    #     определяем по его координатам и координатам центра куба и его размеру
    #     True, если узел граничит с кубом'''
    #     ans = []
    #     answer = False
    #     for vector in self.BASIS:
    #         ans.append(self.isincheck_cube(x + self.incr[0] * vector[0], y + self.incr[1] * vector[1],
    #                                        z + self.incr[2] * vector[2], cx, cy, cz, a))
    #         answer = answer or ans[-1]
    #     return answer

    def nearbordercheck_cube(self, o: tuple, u: tuple) -> float:
        # o - координаты узла, u - базисный вектор
        """Определаяем, граничит ли данный узел с границей куба
         в пределах и направлении заданного вектора (расстояние - в единицах шага решётки).
        Возвращает 0, если узел не граничит в этом направлении, или лежит на границе (расстояние = 0).
        Если граничит - возвращает расстояние (по направлению вектора)
        в пределах расстояния между узлами в этом направлении."""
        cb = self.cube_centre  # координаты центра куба
        a = self.cube_size  # размер куба
        if abs(o[0] - cb[0]) == a / 2 or abs(o[1] - cb[1]) == a / 2 or abs(o[2] - cb[2]) == a / 2:
            return 0

        x1 = o[0] + u[0] * self.incr[0]
        y1 = o[1] + u[1] * self.incr[1]
        z1 = o[2] + u[2] * self.incr[2]

        ans1 = self.isincheck_cube(o[0], o[1], o[2], cb[0], cb[1], cb[2], a)
        ans2 = self.isincheck_cube(x1, y1, z1, cb[0], cb[1], cb[2], a)
        if ans1 == ans2:  # если изначальная точка и ближайшая по базисному
            # вектору - по одну сторону границы, то возвращаем 0
            # if self.m < 1000: # для отлова ошибок принты
            #     self.m += 1
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'*ans1*ans2, end='')
            #     print(f'm={self.m}', ' ' * (4 - len(str(self.m))), f'o=', *o, ' ' * (48 - len(str(o))),
            #           f' u={u} {ans1} {ans2}', ' ' * (18 - len(str(u))), f' x={x1} {y1} {z1}')
            return 0

        # typeansw = [(o[0] < -a / 2 and x1 > -a / 2)]
        # typeansw.append((o[0] > a / 2 and x1 < a / 2))
        # typeansw.append((o[1] < -a / 2 and y1 > -a / 2))
        # typeansw.append((o[1] > a / 2 and y1 < a / 2))
        # typeansw.append((o[2] < -a / 2 and z1 < -a / 2))
        # typeansw.append((o[2] > a / 2 and z1 < a / 2))
        # # ans1 = (o[0] < -a / 2 and x1 > -a / 2)
        # # ans2 = (o[0] < -a / 2 and x1 > -a / 2)
        # #     pointtype_cube = 1
        return 0

    def create_outfile(self):
        """Вызываем все функции, делаем вычисления и записываем в файл"""
        self.get_input()
        self.count_nods_xyz()
        self.x_min = self.figure_centre[0] - self.num_of_nodes[0] * self.incr[0]
        self.y_min = self.figure_centre[1] - self.num_of_nodes[2] * self.incr[1]
        self.z_min = self.figure_centre[2] - self.num_of_nodes[4] * self.incr[2]
        try:
            with open(self.path_out, "w") as file:
                file.write(
                    f'{self.x_min};{self.y_min};{self.z_min}\n')  # координаты
                # левого нижнего узла решётки (минимальные координаты узла по каждому направлению
                # из Ox, Oy, Oz)

                file.write(f'{self.incr[0]};{self.incr[1]};{self.incr[2]}\n')  # шаг решётки по каждому
                # направлению
                file.write(f'{self.Nx};{self.Ny};{self.Nz}\n')
                # количество узлов по каждой оси

                match self.border_type:  # Количество внутренних узлов, и далее через ; количество узлов
                    # для каждой из границ расчётной области
                    # ТАКЖЕ в этом блоке сохраняются все внешние и граничные узлы в виде списка кортежей их координат,
                    # и распределяются по этим группам
                    case 0:  # произвольная граница с заданием по точкам/аналитически
                        pass

                    case 1:  # сфера
                        nums = [0, 0]
                        self.nods_to_write = [[], []]
                        for i in range(self.Nx + 1):
                            x = self.x_min + i * self.incr[0]
                            for j in range(self.Ny + 1):
                                y = self.y_min + j * self.incr[1]
                                for k in range(self.Nz + 1):
                                    z = self.z_min + k * self.incr[2]
                                    if self.isincheck_sphere(x, y, z, self.sphere_centre[0], self.sphere_centre[1],
                                                             self.sphere_centre[2], self.sphere_r):
                                        nums[0] += 1
                                        self.nods_to_write[0].append((i, j, k))
                                    elif self.isbordercheck_sphere(x, y, z, self.sphere_centre[0],
                                                                   self.sphere_centre[1],
                                                                   self.sphere_centre[2], self.sphere_r):
                                        nums[1] += 1
                                        self.nods_to_write[1].append((i, j, k))
                        # Количество внутренних узлов записываем и количество внешних граничных узлов:
                        file.write(f'{nums[0]};{nums[1]}\n')

                    case 2:  # эллипсоид
                        pass

                    case 3:  # куб
                        nums = [0, 0, 0, 0, 0, 0, 0]
                        self.nods_to_write = [[] for _ in range(7)]
                        for i in range(self.Nx + 1):
                            x = self.x_min + i * self.incr[0]
                            for j in range(self.Ny + 1):
                                y = self.y_min + j * self.incr[1]
                                for k in range(self.Nz + 1):
                                    z = self.z_min + k * self.incr[2]
                                    if self.isincheck_cube(x, y, z, self.cube_centre[0], self.cube_centre[1],
                                                           self.cube_centre[2], self.cube_size):
                                        nums[0] += 1
                                        self.nods_to_write[0].append((i, j, k))
                                    elif (x - self.incr[0]) < (self.cube_centre[0] - self.cube_size / 2):
                                        nums[1] += 1
                                        self.nods_to_write[1].append((i, j, k))
                                    elif (x + self.incr[0]) > (self.cube_centre[0] + self.cube_size / 2):
                                        nums[2] += 1
                                        self.nods_to_write[2].append((i, j, k))
                                    elif (y - self.incr[1]) < (self.cube_centre[1] - self.cube_size / 2):
                                        nums[3] += 1
                                        self.nods_to_write[3].append((i, j, k))
                                    elif (y + self.incr[1]) > (self.cube_centre[1] + self.cube_size / 2):
                                        nums[4] += 1
                                        self.nods_to_write[4].append((i, j, k))
                                    elif (z - self.incr[2]) < (self.cube_centre[2] - self.cube_size / 2):
                                        nums[5] += 1
                                        self.nods_to_write[5].append((i, j, k))
                                    elif (z + self.incr[2]) > (self.cube_centre[2] + self.cube_size / 2):
                                        nums[6] += 1
                                        self.nods_to_write[6].append((i, j, k))
                        # Количество внутренних узлов записываем:
                        file.write(f'{nums[0]}')
                        # и количество внешних граничных узлов записываем:
                        for i in range(1, 7):
                            file.write(f';{nums[i]}')
                        file.write('\n')
                        # for i in range(6):
                        #     print(len(self.nods_to_write[i]))
                        # print(self.border[i])

                    case 4:  # параллелепипед
                        pass

                match self.border_type:
                    # Далее всё одинаково:
                    #
                    #     id; i; j; k;
                    #     26 * расстояний (от узла до границы, в пределах одного узла) по 26 заданным направлениям;
                    #     вектор нормали к ближайшей границе (единичный);
                    #     расстояние до ближайшей границы (по вектору нормали будет).
                    #
                    #     Идентификатор узла определяется id = i + (j + k * Ny) * Nx,
                    #     сначала список узлов внутренних, потом по очереди всех границ

                    case 0:  # произвольная граница с заданием по точкам/аналитически
                        pass
                    case 1:  # сфера
                        for n in range(2):
                            # self.m = 0  # счётчик принтов для отлова ошибок
                            # сначала для внутренних узлов считаем и записываем (n=0), затем для границы (n=1):
                            for i, j, k in self.nods_to_write[n]:
                                id_ = i + (j + k * self.Ny) * self.Nx
                                file.write(f'{id_};{i};{j};{k}')
                                x = self.x_min + i * self.incr[0]
                                y = self.y_min + j * self.incr[1]
                                z = self.z_min + k * self.incr[2]
                                # 26 * расстояний по 26 заданным направлениям:
                                for vector in self.BASIS:
                                    file.write(f';{self.nearbordercheck_sphere((x, y, z), vector)}')

                                match n:
                                    case 0: #внутренние узлы (снаружи сферы)
                                        dx = self.sphere_centre[0] - x
                                        dy = self.sphere_centre[1] - y
                                        dz = self.sphere_centre[2] - z
                                        length = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
                                        # вектор нормали к ближайшей границе (единичный):
                                        file.write(f';{dx / length};{dy / length};{dz / length}')
                                        # расстояние до ближайшей границы (по вектору нормали):
                                        file.write(f';{length - self.sphere_r}')
                                        # print(
                                        #     f'hello:    {dx / length};{dy / length};{dz / length};{length - self.sphere_r}',
                                        #     x, y, z)
                                    case 1: #граничные узлы (внутри сферы)
                                        dx = x - self.sphere_centre[0]
                                        dy = y - self.sphere_centre[1]
                                        dz = z - self.sphere_centre[2]
                                        length = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
                                        # вектор нормали к ближайшей границе (единичный):
                                        file.write(f';{dx / length};{dy / length};{dz / length}')
                                        # расстояние до ближайшей границы (по вектору нормали):
                                        file.write(f';{self.sphere_r - length}')
                                        # print(
                                        #     f'hello: {dx / length};{dy / length};{dz / length};{self.sphere_r - length}',
                                        #     x, y, z)

                                file.write(f'\n')

                    case 2:  # эллипсоид
                        pass
                    case 3:  # куб
                        for n in range(7):
                            # self.m = 0  # счётчик принтов для отлова ошибок
                            # сначала для внутренних узлов считаем и записываем (n=0), затем для всех 6 границ (n>=1):
                            for i, j, k in self.nods_to_write[n]:
                                id_ = i + (j + k * self.Ny) * self.Nx
                                file.write(f'{id_};{i};{j};{k}')
                                x = self.x_min + i * self.incr[0]
                                y = self.y_min + j * self.incr[1]
                                z = self.z_min + k * self.incr[2]
                                # 26 * расстояний по 26 заданным направлениям:
                                for vector in self.BASIS:
                                    file.write(f';{self.nearbordercheck_cube((x, y, z), vector)}')

                                # вектор нормали к ближайшей границе (единичный):

                                # расстояние до ближайшей границы (по вектору нормали):

                                file.write(f'\n')
                    case 4:  # параллелепипед
                        pass

        except Exception as error:
            print(f'Ошибка при создании выходного файла:\n{error}')


# match self.border_type:  # zzz
#     case 0:  # произвольная граница с заданием по точкам/аналитически
#         pass
#     case 1:  # сфера
#         pass
#     case 2:  # эллипсоид
#         pass
#     case 3:  # куб
#         pass
#     case 4:  # параллелепипед
#         pass


creator1 = LatticeCreator()
creator1.create_outfile()
# print(creator1.BASIS)
# print(creator1.vect_scalar_mult((1, 2, 3), (1, 2, 3))**0.5)
# brdr = [[(1, 2, 3), (7, 9, 32), (11, 18, 98)], [], [(1, 5, 1)], [], [], [(1,5,5), (7,8,644),(1,5,7)]]
# x = 1
# y = 5
# z = 0
# print(LatticeCreator.isnewnode(brdr, x, y, z))

# print(*creator1.nods_xyz, sep='\n')
# print(creator1.isoutcheck_sphere(7.07106781186,25,15,0,0,0,30))
# print(creator1.isoutcheck_cube(-9.999999999999999,16,14.999999999999999,-5,1,0,30))
