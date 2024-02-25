import numpy as np

from .simulation_curve import one_sample_curve
from .tools import get_duration_from_bound_crossing, get_value_at_duration


def one_sample_one_sided_sprt(x, p0, d, alpha, beta, alternative,
                              initial_curve=None, n_list=None):
    """
    Последовательный анализ в случае одновыборочной задачи
    и односторонней альтернативы

    :param x: массив размера [iter_size, sample_size],
              где каждая строка - значение выборки теста из {0, 1} размера sample_size,
              а iter_size - количество итераций моделирования (тестов)
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование односторонней альтернативы
    :param initial_curve: список длины iter_size из значений
                          логарифмического отношения правдоподобий
                          к моменту применения последовательного анализа
    :param n_list: массив значений прошедшей длительности
    :return: словарь res
             res["duration"] - список длительностей теста
             res["result"] - список результатов теста
             res["result_s"] - список значений S(n) на момент длительности теста
             res["last_curve"] - список значений кривой в последний момент времени
    """

    # Расчёт накопленной суммы S(n) из X(i), i <= n
    x = np.array(x)
    s_list = np.cumsum(x, axis=1)

    # Получение логарифмического отношения правдоподобий
    # и границ для принятия решений
    res = one_sample_curve(s_list=s_list,
                           p0=p0,
                           d=np.abs(d),
                           alpha=alpha,
                           beta=beta,
                           alternative=alternative,
                           n_list=n_list)
    curve = res["curve"]
    low_bound = res["low_bound"]
    high_bound = res["high_bound"]

    # Если заданы значения кривой,
    # корректируем логарифмическое отношение правдоподобий
    if initial_curve is not None:
        curve += initial_curve.reshape(-1, 1)

    # Расчёт индикаторов пересечения границ
    high_bound_crossing_flg = curve > high_bound
    low_bound_crossing_flg = curve < low_bound
    bound_crossing_flg = low_bound_crossing_flg | high_bound_crossing_flg

    # Определение длительности теста
    duration_list = get_duration_from_bound_crossing(bound_crossing_flg)

    # Значение флагов пересечения границы на момент длительности теста
    high_bound_duration_crossing_flg = get_value_at_duration(value_list=high_bound_crossing_flg,
                                                             duration_list=duration_list)
    low_bound_duration_crossing_flg = get_value_at_duration(value_list=low_bound_crossing_flg,
                                                            duration_list=duration_list)

    # Значение S(n), где n - момент длительности теста
    result_s_list = get_value_at_duration(value_list=s_list,
                                          duration_list=duration_list)

    # Определение финального результата теста
    result_list = np.where(high_bound_duration_crossing_flg, 1, 0) \
                  - np.where(low_bound_duration_crossing_flg, 1, 0)

    return {
        "duration": duration_list,
        "result": result_list,
        "result_s": result_s_list,
        "last_curve": curve[:, -1]
    }
