import numpy as np


class QR:
    """Класс QR разложения. Метод получения разложения: метод отражений Хаусхолдера."""
    def transform_truncated(self, A, b=None, machine_zero=1e-8, save_support_vectors=False, verbose=False):
        """
        Метод получения матрицы R. В методе применяется выбор ведущего элемента, что позволяет
        исползовать его для решения переопределенных задач в смысле наименьших квадратов.
        params:
            **b** - вспомогательный столбец, который также нужно вращать, но не нужно занулять(правая часть).

            **machine_zero** - машинный ноль.

            При **save_support_vectors** = False, создается матрица R, иначе на нижней поддиагонале будут
            находится вспомогательные элементы w, необходимые для совершения отражений.

            **verbose** - нужно ли выводить шаги алгоритма.

        return:
            **QR** - матрица R, либо R со вспомогательными эементами w.

            **perm** - список перестановок

            **rank** - ранг матрицы A
        """

        # обьявление списка перестановок, списка норм, ранга матрицы A.
        perm = np.arange(A.shape[1])
        norms = (A ** 2).sum(axis=0)
        rank = A.shape[1]
        
        # Обьединяем A и b (передается как копия), либо копируем A.
        if b is not None:
            QR = np.hstack([A, b.reshape(-1,1)])
        else:
            QR = A.copy()
        
        if verbose:
            print(np.around(QR,6))

        for i in range(rank):
            # Проверяем, остались ли ненулевые столбики в матрице.
            if norms[i:].max() < machine_zero:
                rank = i
                break
            # Выбираем ведущий элемент (с наибольшей нормой).
            lead_col = norms[i:].argmax() + i

            if verbose:
                print('-'*50)
                print(f'Шаг {i}, нормы: {np.around(norms,6)}')
                print(f'Ведущий столбец №{lead_col}')

            # Ставим ведующий элемент на первое место в текущей подматрице.
            perm[[i, lead_col]] = perm[[lead_col, i]]
            QR[:, [i, lead_col]] = QR[:, [lead_col, i]]
            norms[[i, lead_col]] = norms[[lead_col, i]]

            # Совершаем отражение
            self._reflex(QR[i:, i:], 0, inplace=True, save_support_vector=save_support_vectors)

            if verbose:
                print('Применяя отражение, получим следующую матрицу:')
                print(np.around(QR,6))

            # Пересчитываем нормы подматрицы (используя свойство сохранения
            # нормы у ортогонального преобразования).
            norms[i+1:] = norms[i+1:] - QR[i, i+1: rank]**2
        return QR, perm, rank
    
    def transform(self, A, machine_zero=1e-8):
        """Получение QR разложения с использованием ведущего элемента.
        params:
            **A** - матрица, для которой ищется разложение

        return:
            **(Q,R)** - матрицы Q и R

            **permutes** - перестановки столбцов

            **rank** - ранг матрицы A"""
        QR_truncated, permutes, rank = self.transform_truncated(
                    A, 
                    machine_zero=machine_zero,
                    save_support_vectors=True)
        Q = self._build_Q_matrix(QR_truncated, rank=rank)

        # Очистим матрицу QR_truncated от поддиагонаьных элементов.
        R = np.zeros(QR_truncated.shape)
        for i in range(rank):
            R[i, i:] = QR_truncated[i, i:].copy()
        return (Q, R), permutes, rank
    
    def _build_Q_matrix(self, W, rank):
        """
        Получение ортогональной матрицы Q.
        Это наиболее точная но и наиболее затратная версия.
        Просто выполняется формирование и перемножение всех матриц
        поворотов
        params:
            **W** - матрица R, с сохранненными векторами w на поддиагоналях.

        return:
            **Q.T** - ортогональная матрица Q.
        """
        Q = np.eye(W.shape[0])
        r = W.shape[0]
        for i in range(rank):
            w = W[i:, i].copy()
            w[0] = 1
            Alpha = w.reshape(-1, 1).dot(w.reshape(1, -1))
            beta = 2 / w.dot(w)
            # Вначале обьявляем матрицу V как единичную
            V = np.eye(W.shape[0])
            # Меняем минор соответствующего порядка.
            V[i:, i:] = np.eye(r) - beta * Alpha
            r -= 1
            Q = V.dot(Q)
        return Q.T
    
    def OLS(self, A, b, machine_zero=1e-8, verbose=False):
        """Реализация метода наименьших квадратов для ||Ax-b||
        с помощью QR разложения, с выбором
        ведущего элемента.
        params:
            **A** - матрица

            **b** - вектор правых частей

            **machine_zero** - машинный ноль

            **verbose** 
        
        return:
            **x** - главный вектор решения

            **residual_norm** - норма невязки."""
        # Находим отраженную матрицу R и отраженные правые части.
        R, permutes, rank = self.transform_truncated(
            A=A, 
            b=b, 
            machine_zero=machine_zero,
            verbose=verbose)
    
        R_ = R[:rank, :rank]
        b_ = R[:rank, -1]

        # Находим норму невязки решения, как норму
        # элементов правой части, ниже ведущих.
        residual_norm = np.linalg.norm(R[rank:, -1], ord=2)

        # Вычисляем вектор решения ведущей системы.
        x_ = self._reverse_motion(R_, b_)
        # Добавляем нули, получая таким образом главный вектор решения.
        x = np.append(x_, np.zeros(A.shape[1] - rank))
        # Переставляем элементы решения в верном порядке.
        x = x[permutes]
        return x, residual_norm
        
        
    def solve(self, A, b):
        """Решение определенной СЛАУ **Ax=b**"""
        R_b = self.transform_truncated(A, b)
        R = R_b[:, :-1]
        b_ = R_b[:, -1]
        x = self._reverse_motion(R, b_)
        return x
    
    def _reflex(self, A, column, inplace=False, save_support_vector=False, w=None):
        """
        Отражение матрицы A вдоль столбика column.
        

        params:
            **A** - матрица, в которой нужно отразить столбик

            **column** - номер столбика, вдоль которого делается отражение 

            **inplace** - если inplace=True, то входная 
                        матрица A становится отраженной.

            **save_support_vector** - нужно ли сохранять вспомогательные векторы

            **w** - если необходимо совершить отражение по заданному вспомогательному вектору

        return:
            **A** - отраженная матрица.
        """
        if w is None:
            w = self._get_support_vector(A[:, column])
        if save_support_vector:
            w = w / w[0]
        # Вспомогательные величины.
        b = (A.T).dot(w)
        beta = 2 / w.dot(w)
        
        if not inplace:
            A = A.copy()
        
        # Применяем отражение.
        A[:, :] = A - beta * (w.reshape(-1, 1).dot(b.reshape(1,-1)))

        if save_support_vector:
            A[1:, column] = w[1:]

        return A
        
    
    def _get_support_vector(self, x):
        """
        Получение вспомогательного
        вектора для отражения. 
        params:
            **x** - вектор, с помощью которого получаем вспомогательный

        return:
            **w** - вспомогательный вектор для отражения.
        """
        gamma = np.linalg.norm(x, ord=2)

        # Погрешность меньше, если скадываются элементы с одинаковыми знаками.
        if x[0] < 0:
            gamma = -gamma
        
        w = x.copy()
        w[0] = w[0] + gamma
        return w
    
    def _reverse_motion(self, A, b):
        """Обратный ход решения. По сути, решение СЛАУ **Ax=b**, в случае, когда
        A - треугольная.
        params:
            **A** - треугольная матрица
            **b** - вектор правых часте
            
        return:
            **x** - вектор решений."""
        x = np.zeros(b.shape)
        for i in range(b.shape[0])[::-1]:
            left_part = A[i, i+1:].dot(x[i+1:])
            x[i] = (b[i] - left_part) / A[i, i]
        return x

if __name__ == '__main__':
    A = np.array([
        [1,2,20],
        [4,5,155],
        [7,8,10],
        # [2,3,4]
    ], dtype='float64')
    b = np.array([
        -1,
        2,
        3,
        # 1
    ], dtype='float64')

    print(f'A:\n{A}')
    print(f'b:\n{b}')

    qr = QR()
    x, r = qr.OLS(A, b, verbose=True)

    print(f'x:\n{x}')
    print('Норма невязки =', r)
    print('Ax =', np.around(A.dot(x),6))

    (Q, R), permutes, rank = qr.transform(A)
    P = np.eye(A.shape[1])[:, permutes]
    print(f'Матрица Q в явном виде:\n{Q}')
    print(f'Матрица перестановок П:\n{P}')
    print('AП:')
    print(np.around(A.dot(P),6))
    print('QR:')
    print(np.around(Q.dot(R),6))

    print('#'*100)
    print('Все тоже самое, только для индивидуального задания:')


    A = np.array([
        [0, 1, 1, 0, 1],
        [1, 0, -1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, -1, 1],
        [0, 0, 1, 0, 1],
        [0, 1, 1, 0, -1]
    ], dtype='float64')
    b = np.array([
        1,
        1,
        0,
        0,
        1,
        0
    ], dtype='float64')

    print(f'A:\n{A}')
    print(f'b:\n{b}')

    x, r = qr.OLS(A, b, verbose=False)

    print(f'x:\n{x}')
    print('Норма невязки =', r)
    print('Ax =', np.around(A.dot(x),6))

    (Q, R), permutes, rank = qr.transform(A)
    P = np.eye(A.shape[1])[:, permutes]
    print(f'Матрица Q в явном виде:\n{Q}')
    print(f'Матрица перестановок П:\n{P}')
    print('AП:')
    print(np.around(A.dot(P),6))
    print('QR:')
    print(np.around(Q.dot(R),6))