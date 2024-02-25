import numpy as np

from binary.checking.one_sample_two_sided_sprt import one_sample_two_sided_sprt
from binary.checking.tools import transform_two_sample_one_sided_mde, get_value_at_duration


def two_sample_two_sided_sprt(x, y, p0, d, alpha, beta,
                              greater_initial_curve=None, less_initial_curve=None,
                              greater_stop_flg=None, less_stop_flg=None):
    """
    Последовательный анализ в случае двухвыборочной задачи
    и двусторонней альтернативы

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
    :param greater_initial_curve: список длины iter_size из значений
                                  логарифмического отношения правдоподобий
                                  к моменту применения последовательного анализа
                                  при проверке гипотезы p0 против p0 + d приостановлена
    :param less_initial_curve: список длины iter_size из значений
                               логарифмического отношения правдоподобий
                               к моменту применения последовательного анализа
                               при проверке гипотезы p0 - d против p0 приостановлена
    :param greater_stop_flg: список длины iter_size из флагов того,
                             что в конкретном тесте проверка гипотезы p0 против p0 + d приостановлена
    :param less_stop_flg: список длины iter_size из флагов того,
                          что в конкретном тесте проверка гипотезы p0 - d против p0 приостановлена
    :return: словарь res
             res["duration"] - список длительностей теста
             res["result"] - список результатов теста
             res["result_x_s"] - список значений S(n) для первой выборки на момент длительности теста
             res["result_y_s"] - список значений S(n) для второй выборки на момент длительности теста
             res["greater_last_curve"] - список значений кривой в последний момент времени
                                         при проверке гипотезы p0 против p0 + d приостановлена
             res["less_last_curve"] - список значений кривой в последний момент времени
                                      при проверке гипотезы p0 - d против p0 приостановлена
             res["greater_stop"] - список флагов того, что в конкретном тесте
                                   проверка гипотезы p0 против p0 + d приостановлена
             res["less_stop"] - список флагов того, что в конкретном тесте
                                проверка гипотезы p0 - d против p0 приостановлена
    """

    # Расчёт накопленной суммы S(n) из X(i) и Y(i), i <= n
    x = np.array(x)
    x_s_list = np.cumsum(x, axis=1)
    y = np.array(y)
    y_s_list = np.cumsum(y, axis=1)

    # Определение параметров одновыборочного последовательного теста
    p0_transformed = 1 / 2
    d_low_transformed = transform_two_sample_one_sided_mde(p0, d, alternative="less")
    d_high_transformed = transform_two_sample_one_sided_mde(p0, d, alternative="greater")

    # Преобразование двувыборочной задачи к одновыборочной
    z = x * (1 - y)
    n_list = np.cumsum(np.where(x != y, 1, 0), axis=1)

    # Вычисление результатов последовательного результата одновыборочной задачи
    one_sample_res = one_sample_two_sided_sprt(z, p0_transformed,
                                               [d_low_transformed, d_high_transformed],
                                               alpha, beta,
                                               greater_initial_curve=greater_initial_curve,
                                               less_initial_curve=less_initial_curve,
                                               greater_stop_flg=greater_stop_flg,
                                               less_stop_flg=less_stop_flg,
                                               n_list=n_list)

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
        "greater_last_curve": one_sample_res["greater_last_curve"],
        "less_last_curve": one_sample_res["less_last_curve"],
        "greater_stop": one_sample_res["greater_stop"],
        "less_stop": one_sample_res["less_stop"]
    }
