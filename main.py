from itertools import chain


class LatticeCreator:
    BASIS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, 1, 0], [1, -1, 0],
             [-1, -1, 0],
             [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
             [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]

    def __init__(self, path_in='input.txt', path_out='output.txt'):
        self.path_in = path_in
        self.path_out = path_out

    def get_input(self):
        '''достаём все параметры из входного файла'''
        try:
            with open(self.path_in, encoding='utf-8') as file:
                lines = file.readlines()
                self.minmaxcoord = tuple(map(float, lines[0].split(';')))  # крайние
                # координаты (мин, макс) области вычислений по осям координат
                print('крайние координаты (мин, макс) области вычислений: ', self.minmaxcoord)
                self.incr = tuple(map(float, lines[1].split(';')))  # шаги решётки по осям координат
                print('шаги решётки по осям координат: ', self.incr)
                self.border_type = int(lines[2].strip())  # тип фигуры для задания границы
                # print(self.border_type, type(self.border_type))

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
                self.create_nods_xyz()
        except:
            print('Неверный формат входного файла')

    def create_nods_xyz(self):
        '''Создаём список (по осям) списков координат всех узлов
        решётки для дальнейших проверок и вычислений'''
        self.num_of_nodes = []
        for n in range(6):  # вычисляем и складываем количество узлов
            # по разные стороны центра фигуры границы, для каждой оси.
            # На данный момент узел решётки совпадает с центром фигуры границы
            # и откладывается от него до границ области вычислений.
            # При этом если шаги узлов решётки кратны расстояниям, получаем много узлов решётки,
            # расположенных ровно в границе области вычислений! Уточнить нормально ли это!!!!!
            # print(n % 2, end=' ')
            if not n % 2:
                self.num_of_nodes.append(
                    int(abs((self.minmaxcoord[n] - self.figure_centre[n // 2]) / self.incr[n // 2])))
            else:
                self.num_of_nodes.append(
                    int(abs((self.minmaxcoord[n] - self.figure_centre[n // 2]) / self.incr[n // 2])))
        # print(self.num_of_nodes)
        a = chain(range(self.num_of_nodes[0] * (-1), 0), range(self.num_of_nodes[1] + 1))
        b = chain(range(self.num_of_nodes[2] * (-1), 0), range(self.num_of_nodes[3] + 1))
        c = chain(range(self.num_of_nodes[4] * (-1), 0), range(self.num_of_nodes[5] + 1))
        # print(a, b, c, sep='\n')
        self.nods_xyz = [[], [], []]
        for nx in a:
            self.nods_xyz[0].append(nx * self.incr[0] + self.figure_centre[0])
        for ny in b:
            self.nods_xyz[1].append(ny * self.incr[1] + self.figure_centre[1])
        for nz in c:
            self.nods_xyz[2].append(nz * self.incr[2] + self.figure_centre[2])
        return self.nods_xyz

    @staticmethod
    def isoutcheck_sphere(x, y, z, cx, cy, cz, r):
        ''' определаяем внешний/внутренний ли узел по
        его координатам и координатам центра сферы и её радиуса
        True, если узел внутренний (за пределами границы, внутри расчётной области)'''
        return ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5 >= r

    def isbordercheck_sphere(self, x, y, z, cx, cy, cz, r):
        '''Определаяем, граничит ли внешний узел с границей сферы (внутренним узлом)
        хотя бы по одной своей координате; определяем по его координатам и координатам
        центра сферы и её радиуса
        True, если узел граничит со сферой'''
        ans1 = self.isoutcheck_sphere(x + self.incr[0], y, z, cx, cy, cz, r)
        ans2 = self.isoutcheck_sphere(x - self.incr[0], y, z, cx, cy, cz, r)
        ans3 = self.isoutcheck_sphere(x, y + self.incr[1], z, cx, cy, cz, r)
        ans4 = self.isoutcheck_sphere(x, y - self.incr[1], z, cx, cy, cz, r)
        ans5 = self.isoutcheck_sphere(x, y, z + self.incr[2], cx, cy, cz, r)
        ans6 = self.isoutcheck_sphere(x, y, z - self.incr[2], cx, cy, cz, r)
        return ans1 or ans2 or ans3 or ans4 or ans5 or ans6

    @staticmethod
    def isoutcheck_cube(x, y, z, cx, cy, cz, a):
        ''' определаяем внутренний/внешний ли узел по
        его координатам и координатам центра куба и размеру его грани
        True, если узел внутренний (за пределами границы, внутри расчётной области)'''
        return abs(x - cx) >= a / 2 or abs(y - cy) >= a / 2 or abs(z - cz) >= a / 2

    # def isnewnode(self, x, y, z):
    #     '''Функция проверяет (на отсутствие в списке узлов) внешний узел
    #     (узел кубической границы) для всех 6 границ расчётных областей куба'''
    #     for i in range(6):
    #         if (x, y, z) in self.border[i]:
    #             return False
    #     return True

    def isbordercheck_cube(self, x, y, z, cx, cy, cz, a):
        '''Определаяем, граничит ли внешний узел с границей куба (внутренним узлом)
        хотя бы по одной своей координате;
        определяем по его координатам и координатам центра куба и его размеру
        True, если узел граничит с кубом'''
        ans1 = self.isoutcheck_cube(x + self.incr[0], y, z, cx, cy, cz, a)
        ans2 = self.isoutcheck_cube(x - self.incr[0], y, z, cx, cy, cz, a)
        ans3 = self.isoutcheck_cube(x, y + self.incr[1], z, cx, cy, cz, a)
        ans4 = self.isoutcheck_cube(x, y - self.incr[1], z, cx, cy, cz, a)
        ans5 = self.isoutcheck_cube(x, y, z + self.incr[2], cx, cy, cz, a)
        ans6 = self.isoutcheck_cube(x, y, z - self.incr[2], cx, cy, cz, a)
        return ans1 or ans2 or ans3 or ans4 or ans5 or ans6
        # return x + self.incr[0]> abs(cx)

    def create_outfile(self):
        '''Вызываем все функции, делаем вычисления и записываем в файл'''
        self.get_input()
        self.create_nods_xyz()
        try:
            with open(self.path_out, "w") as file:
                file.write(
                    f'{self.nods_xyz[0][0]};{self.nods_xyz[1][0]};{self.nods_xyz[2][0]}\n')  # координаты
                # левого нижнего узла решётки (минимальные координаты узла по каждому направлению
                # из Ox, Oy, Oz)

                file.write(f'{self.incr[0]};{self.incr[1]};{self.incr[2]}\n')  # шаг решётки по каждому
                # направлению

                file.write(f'{len(self.nods_xyz[0])};{len(self.nods_xyz[1])};{len(self.nods_xyz[2])}\n')
                # количество узлов по каждой оси

                match self.border_type:  # Количество внутренних узлов, и далее через ; количество узлов
                    # для каждой из границ расчётной области
                    case 0:  # произвольная граница с заданием по точкам/аналитически
                        pass

                    case 1:  # сфера
                        num_out = 0
                        num_in = 0
                        for x in self.nods_xyz[0]:
                            for y in self.nods_xyz[1]:
                                for z in self.nods_xyz[2]:
                                    if self.isoutcheck_sphere(x, y, z, self.sphere_centre[0], self.sphere_centre[1],
                                                              self.sphere_centre[2], self.sphere_r):
                                        num_out += 1
                                    elif self.isbordercheck_sphere(x, y, z, self.sphere_centre[0],
                                                                   self.sphere_centre[1],
                                                                   self.sphere_centre[2], self.sphere_r):
                                        num_in += 1
                        # Количество внутренних узлов записываем:
                        file.write(f'{num_out};{num_in}')

                    case 2:  # эллипсоид
                        pass

                    case 3:  # куб
                        num_out = 0
                        num_in = [0, 0, 0, 0, 0, 0]
                        self.border = [[] for _ in range(6)]
                        for x in self.nods_xyz[0]:
                            for y in self.nods_xyz[1]:
                                for z in self.nods_xyz[2]:
                                    if self.isoutcheck_cube(x, y, z, self.cube_centre[0], self.cube_centre[1],
                                                            self.cube_centre[2], self.cube_size):
                                        num_out += 1
                                    elif (x - self.incr[0]) < (self.cube_centre[0] - self.cube_size / 2):
                                        num_in[0] += 1
                                        # if self.isnewnode(x, y, z):
                                        #     self.border[0].append((x, y, z))
                                        self.border[0].append((x, y, z))
                                    elif (x + self.incr[0]) > (self.cube_centre[0] + self.cube_size / 2):
                                        num_in[1] += 1
                                        # if self.isnewnode(x, y, z):
                                        #     self.border[1].append((x, y, z))
                                        self.border[1].append((x, y, z))
                                    elif (y - self.incr[1]) < (self.cube_centre[1] - self.cube_size / 2):
                                        num_in[2] += 1
                                        # if self.isnewnode(x, y, z):
                                        #     self.border[2].append((x, y, z))
                                        self.border[2].append((x, y, z))
                                    elif (y + self.incr[1]) > (self.cube_centre[1] + self.cube_size / 2):
                                        num_in[3] += 1
                                        # if self.isnewnode(x, y, z):
                                        #     self.border[3].append((x, y, z))
                                        self.border[3].append((x, y, z))
                                    elif (z - self.incr[2]) < (self.cube_centre[2] - self.cube_size / 2):
                                        num_in[4] += 1
                                        # if self.isnewnode(x, y, z):
                                        #     self.border[4].append((x, y, z))
                                        self.border[4].append((x, y, z))
                                    elif (z + self.incr[2]) > (self.cube_centre[2] + self.cube_size / 2):
                                        num_in[5] += 1
                                        # if self.isnewnode(x, y, z):
                                        #     self.border[5].append((x, y, z))
                                        self.border[5].append((x, y, z))
                                    # Количество внутренних узлов записываем:
                        file.write(f'{num_out}')
                        for i in range(6):
                            file.write(f';{num_in[i]}')
                        for i in range(6):
                            print(len(self.border[i]))
                            print(self.border[i])

                    case 4:  # параллелепипед
                        pass

                match self.border_type:  # и далее всё одинаково:
                    # id; i; j; k; 26 * расстояний (между чем? От узла до границы, в пределах одного узла) по
                    # направлению (позже); вектор нормали к ближайшей границе (что это и как считается?);
                    # расстояние до ближайшей границы (по вектору нормали?да)
                    # идентификатор узла определяется id = i + (j + k * Ny) *Nx,
                    # сначала список узлов внутренних, потом по очереди всех границ
                    case 0:  # произвольная граница с заданием по точкам/аналитически
                        pass
                    case 1:  # сфера
                        pass
                    case 2:  # эллипсоид
                        pass
                    case 3:  # куб
                        pass
                    case 4:  # параллелепипед
                        pass

        except:
            print('Ошибка при создании выходного файла')


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
# brdr = [[(1, 2, 3), (7, 9, 32), (11, 18, 98)], [], [(1, 5, 1)], [], [], [(1,5,5), (7,8,644),(1,5,7)]]
# x = 1
# y = 5
# z = 0
# print(LatticeCreator.isnewnode(brdr, x, y, z))

# print(*creator1.nods_xyz, sep='\n')
# print(creator1.isoutcheck_sphere(7.07106781186,25,15,0,0,0,30))
# print(creator1.isoutcheck_cube(-9.999999999999999,16,14.999999999999999,-5,1,0,30))
