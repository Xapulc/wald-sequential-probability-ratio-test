import numpy as np

from binary.checking.one_sample_one_sided_sprt import one_sample_one_sided_sprt
from binary.checking.tools import transform_two_sample_one_sided_mde, get_value_at_duration


def two_sample_one_sided_sprt(x, y, p0, d, alpha, beta, alternative,
                              initial_curve=None):
    """
    Последовательный анализ в случае двухвыборочной задачи
    и односторонней альтернативы

    :param x: массив размера [iter_size, sample_size],
              где каждая строка - значение первой выборки теста из {0, 1} размера sample_size,
              а iter_size - количество итераций моделирования (тестов)
    :param y: массив размера [iter_size, sample_size],
              где каждая строка - значение второй выборки теста из {0, 1} размера sample_size,
              а iter_size - количество итераций моделирования (тестов)
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование односторонней альтернативы
    :param initial_curve: список длины iter_size из значений
                          логарифмического отношения правдоподобий
                          к моменту применения последовательного анализа
    :return: словарь res
             res["duration"] - список длительностей теста
             res["result"] - список результатов теста
             res["result_x_s"] - список значений S(n) для первой выборки на момент длительности теста
             res["result_y_s"] - список значений S(n) для второй выборки на момент длительности теста
             res["last_curve"] - список значений кривой в последний момент времени
    """

    # Расчёт накопленной суммы S(n) из X(i) и Y(i), i <= n
    x = np.array(x)
    x_s_list = np.cumsum(x, axis=1)
    y = np.array(y)
    y_s_list = np.cumsum(y, axis=1)

    # Определение параметров одновыборочного последовательного теста
    p0_transformed = 1 / 2
    d_transformed = transform_two_sample_one_sided_mde(p0, d, alternative=alternative)

    # Преобразование двувыборочной задачи к одновыборочной
    z = x * (1 - y)
    n_list = np.cumsum(np.where(x != y, 1, 0), axis=1)

    # Вычисление результатов последовательного результата одновыборочной задачи
    one_sample_res = one_sample_one_sided_sprt(z, p0_transformed, d_transformed, alpha, beta, alternative,
                                               initial_curve=initial_curve, n_list=n_list)

    # Значение S(n), где n - момент длительности теста
    result_x_s_list = get_value_at_duration(value_list=x_s_list,
                                            duration_list=one_sample_res["duration"])
    result_y_s_list = get_value_at_duration(value_list=y_s_list,
                                            duration_list=one_sample_res["duration"])

    return {
        "duration": one_sample_res["duration"],
        "result": one_sample_res["result"],
        "result_x_s": result_x_s_list,
        "result_y_s": result_y_s_list,
        "last_curve": one_sample_res["last_curve"]
    }
