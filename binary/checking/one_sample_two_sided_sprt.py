import numpy as np

from .simulation_curve import one_sample_curve
from .tools import get_duration_from_bound_crossing, get_value_at_duration


def one_sample_two_sided_sprt(x, p0, d, alpha, beta,
                              greater_initial_curve=None, less_initial_curve=None,
                              greater_stop_flg=None, less_stop_flg=None):
    """
    Последовательный анализ в случае одновыборочной задачи
    и двусторонней альтернативы

    :param x: массив размера [iter_size, sample_size],
              где каждая строка - значение выборки теста из {0, 1} размера sample_size,
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
             res["result_s"] - список значений S(n) на момент длительности теста
             res["greater_last_curve"] - список значений кривой в последний момент времени
                                         при проверке гипотезы p0 против p0 + d приостановлена
             res["less_last_curve"] - список значений кривой в последний момент времени
                                      при проверке гипотезы p0 - d против p0 приостановлена
             res["greater_stop"] - список флагов того, что в конкретном тесте
                                   проверка гипотезы p0 против p0 + d приостановлена
             res["less_stop"] - список флагов того, что в конкретном тесте
                                проверка гипотезы p0 - d против p0 приостановлена
    """

    # Расчёт накопленной суммы S(n) из X(i), i <= n
    x = np.array(x)
    s_list = np.cumsum(x, axis=1)

    # Получение логарифмического отношения правдоподобий
    # и границ для принятия решений
    res = one_sample_curve(s_list=s_list,
                           p0=p0,
                           d=np.abs(d),
                           alpha=alpha/2,
                           beta=beta,
                           alternative="greater")
    greater_curve = res["curve"]
    greater_low_bound = res["low_bound"]
    greater_high_bound = res["high_bound"]

    res = one_sample_curve(s_list=s_list,
                           p0=p0,
                           d=np.abs(d),
                           alpha=alpha/2,
                           beta=beta,
                           alternative="less")
    less_curve = res["curve"]
    less_low_bound = res["low_bound"]
    less_high_bound = res["high_bound"]

    # Если заданы значения кривой,
    # корректируем логарифмическое отношение правдоподобий
    if greater_initial_curve is not None:
        greater_curve += greater_initial_curve.reshape(-1, 1)

    if less_initial_curve is not None:
        less_curve += less_initial_curve.reshape(-1, 1)

    # Расчёт индикаторов пересечения границ
    greater_high_bound_crossing_flg = greater_curve > greater_high_bound
    greater_low_bound_crossing_flg = greater_curve < greater_low_bound
    greater_bound_crossing_flg = greater_high_bound_crossing_flg | greater_low_bound_crossing_flg

    less_high_bound_crossing_flg = less_curve > less_high_bound
    less_low_bound_crossing_flg = less_curve < less_low_bound
    less_bound_crossing_flg = less_high_bound_crossing_flg | less_low_bound_crossing_flg

    # Определение длительности теста
    greater_duration_list = get_duration_from_bound_crossing(greater_bound_crossing_flg)
    less_duration_list = get_duration_from_bound_crossing(less_bound_crossing_flg)

    # Значение флагов пересечения границы на момент длительности теста
    greater_high_bound_duration_crossing_flg = get_value_at_duration(value_list=greater_high_bound_crossing_flg,
                                                                     duration_list=greater_duration_list)
    greater_low_bound_duration_crossing_flg = get_value_at_duration(value_list=greater_low_bound_crossing_flg,
                                                                    duration_list=greater_duration_list)

    less_high_bound_duration_crossing_flg = get_value_at_duration(value_list=less_high_bound_crossing_flg,
                                                                  duration_list=less_duration_list)
    less_low_bound_duration_crossing_flg = get_value_at_duration(value_list=less_low_bound_crossing_flg,
                                                                 duration_list=less_duration_list)

    # Если проверка гипотез по некоторой логике была остановлена ранее,
    # но при этом тест до сих пор длиться, учитываем это
    if greater_stop_flg is not None:
        greater_high_bound_duration_crossing_flg = greater_high_bound_duration_crossing_flg \
                                                   & (~greater_stop_flg)
        greater_low_bound_duration_crossing_flg = greater_low_bound_duration_crossing_flg \
                                                  | greater_stop_flg
        greater_duration_list = np.where(greater_stop_flg, 0, greater_duration_list)

    if less_stop_flg is not None:
        less_high_bound_duration_crossing_flg = less_high_bound_duration_crossing_flg \
                                                | less_stop_flg
        less_low_bound_duration_crossing_flg = less_low_bound_duration_crossing_flg \
                                               & (~less_stop_flg)
        less_duration_list = np.where(less_stop_flg, 0, less_duration_list)

    # Обновляем логику определения остановленных проверок гипотез
    greater_stop_flg = (greater_stop_flg if greater_stop_flg is not None else False) \
                       | greater_bound_crossing_flg.any(axis=1)
    less_stop_flg = (less_stop_flg if less_stop_flg is not None else False) \
                    | less_bound_crossing_flg.any(axis=1)

    # Определяем значения длительности и результата теста по-дефолту
    # как в случае, если тест ещё не завершён
    duration_list = x.shape[1]
    result_list = 0

    # Случай, когда длительность проверки гипотезы p0 против p0 + d
    # не больше длительности проверки гипотезы p0 - d против p0
    greater_min_duration_flg = greater_duration_list <= less_duration_list

    # Определяем значения длительности и результата теста в случае,
    # когда есть пересечение верхней границы для p0 + d
    duration_list = np.where(greater_min_duration_flg
                             & greater_high_bound_duration_crossing_flg,
                             greater_duration_list, duration_list)
    result_list = np.where(greater_min_duration_flg
                           & greater_high_bound_duration_crossing_flg,
                           1, result_list)

    # Определяем значения длительности и результата теста в случае,
    # когда не было пересечения верхней границы для p0 + d,
    # но затем было пересечение нижней границы для p0 - d
    duration_list = np.where(greater_min_duration_flg
                             & (~greater_high_bound_duration_crossing_flg)
                             & less_low_bound_duration_crossing_flg,
                             less_duration_list, duration_list)
    result_list = np.where(greater_min_duration_flg
                           & (~greater_high_bound_duration_crossing_flg)
                           & less_low_bound_duration_crossing_flg,
                           1, result_list)

    # Определяем значения длительности и результата теста в случае,
    # когда было пересечение границы для p0 в обоих случаях
    duration_list = np.where(greater_min_duration_flg
                             & greater_low_bound_duration_crossing_flg
                             & less_high_bound_duration_crossing_flg,
                             less_duration_list, duration_list)
    result_list = np.where(greater_min_duration_flg
                           & greater_low_bound_duration_crossing_flg
                           & less_high_bound_duration_crossing_flg,
                           -1, result_list)

    # Случай, когда длительность проверки гипотезы p0 против p0 + d
    # не меньше длительности проверки гипотезы p0 - d против p0
    less_min_duration_flg = greater_duration_list >= less_duration_list

    # Определяем значения длительности и результата теста в случае,
    # когда есть пересечение нижней границы для p0 - d
    duration_list = np.where(less_min_duration_flg
                             & less_low_bound_duration_crossing_flg,
                             less_duration_list, duration_list)
    result_list = np.where(less_min_duration_flg
                           & less_low_bound_duration_crossing_flg,
                           1, result_list)

    # Определяем значения длительности и результата теста в случае,
    # когда не было пересечения нижней границы для p0 - d,
    # но затем было пересечение верхней границы для p0 + d
    duration_list = np.where(less_min_duration_flg
                             & (~less_low_bound_duration_crossing_flg)
                             & greater_high_bound_duration_crossing_flg,
                             greater_duration_list, duration_list)
    result_list = np.where(less_min_duration_flg
                           & (~less_low_bound_duration_crossing_flg)
                           & greater_high_bound_duration_crossing_flg,
                           1, result_list)

    # Определяем значения длительности и результата теста в случае,
    # когда было пересечение границы для p0 в обоих случаях
    duration_list = np.where(less_min_duration_flg
                             & less_high_bound_duration_crossing_flg
                             & greater_low_bound_duration_crossing_flg,
                             greater_duration_list, duration_list)
    result_list = np.where(less_min_duration_flg
                           & less_high_bound_duration_crossing_flg
                           & greater_low_bound_duration_crossing_flg,
                           -1, result_list)

    # Значение S(n), где n - момент длительности теста
    result_s_list = get_value_at_duration(value_list=s_list,
                                          duration_list=duration_list)

    return {
        "duration": duration_list,
        "result": result_list,
        "result_s": result_s_list,
        "greater_last_curve": greater_curve[:, -1],
        "less_last_curve": less_curve[:, -1],
        "greater_stop": greater_stop_flg,
        "less_stop": less_stop_flg
    }
