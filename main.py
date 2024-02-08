from itertools import chain
class LatticeCreator:
    BASIS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, 1, 0], [1, -1, 0],
             [-1, -1, 0],
             [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
             [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]

    def __init__(self, path='input.txt'):
        self.path = path

    def get_input(self):  # достаём все параметры из входного файла
        try:
            with open(self.path, encoding='utf-8') as file:
                lines = file.readlines()
                self.minmaxcoord = tuple(map(float, lines[0].split(';')))  # крайние
                # координаты (мин, макс) области вычислений по осям координат
                print(self.minmaxcoord)
                self.incr = tuple(map(float, lines[1].split(';')))  # шаги решётки по осям координат
                print(self.incr)
                self.border_type = int(lines[2].strip())  # тип фигуры для задания границы
                print(self.border_type, type(self.border_type))
                match self.border_type:  # вытаскиваем параметры для каждого типа фигуры
                    case 0:  # произвольная граница с заданием по точкам/аналитически
                        pass
                    case 1:  # сфера
                        t = tuple(map(float, lines[3].split(';')))
                        self.sphere_centre, self.sphere_r = t[:-1], t[-1]  # координаты центра сферы и радиуса
                        print(self.sphere_centre, self.sphere_r)
                    case 2:  # эллипсоид
                        pass
                    case 3:  # куб
                        pass
                    case 4:  # параллелепипед
                        pass
                self.create_nods_xyz()
        except:
            print('Неверный формат входного файла')

    def create_nods_xyz(self):  # создаём список по осям списков координат всех узлов
        # решётки для дальнейших проверок и вычислений
        # ПОКА ДЛЯ СФЕРЫ ПАРАМЕТРЫ НАПРЯМУЮ ИСПОЛЬЗУЮТСЯ!!!
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
                    int(abs((self.minmaxcoord[n] - self.sphere_centre[n // 2]) / self.incr[n // 2])))
            else:
                self.num_of_nodes.append(
                    int(abs((self.minmaxcoord[n] - self.sphere_centre[n // 2]) / self.incr[n // 2])))
        # print(self.num_of_nodes)
        a = chain(range(self.num_of_nodes[0]*(-1),0), range(self.num_of_nodes[1]+1))
        b = chain(range(self.num_of_nodes[2] * (-1), 0), range(self.num_of_nodes[3] + 1))
        c = chain(range(self.num_of_nodes[4] * (-1), 0), range(self.num_of_nodes[5] + 1))
        # print(a, b, c, sep='\n')
        self.nods_xyz = [[], [], []]
        for nx in a:
            self.nods_xyz[0].append(nx*self.incr[0]+self.sphere_centre[0])
        for ny in b:
            self.nods_xyz[1].append(ny*self.incr[1]+self.sphere_centre[1])
        for nz in c:
            self.nods_xyz[2].append(nz*self.incr[2]+self.sphere_centre[2])
        # for nods_xyz in self.nods_xyz:
        #     print(*nods_xyz, sep='\n')
        # return self.nods_xyz

        # match self.border_type:  # вытаскиваем параметры для каждого типа фигуры
        #     case 0:  # произвольная граница с заданием по точкам/аналитически
        #         pass
        #     case 1:  # сфера
        #         t = tuple(map(float, lines[3].split(';')))
        #         self.sphere_centre, self.sphere_r = t[:-1], t[-1]  # координаты центра сферы и радиуса
        #         print(self.sphere_centre, self.sphere_r)
        #     case 2:  # эллипсоид
        #         pass
        #     case 3:  # куб
        #         pass
        #     case 4:  # параллелепипед
        #         pass
        # self.create_nods


creator1 = LatticeCreator()
creator1.get_input()
print(*creator1.nods_xyz, sep='\n')
