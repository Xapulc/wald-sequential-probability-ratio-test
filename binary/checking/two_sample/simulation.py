import numpy as np
from scipy.stats import bernoulli

from binary.checking.one_sample_one_sided_sprt import one_sample_one_sided_sprt
from binary.checking.one_sample_two_sided_sprt import one_sample_two_sided_sprt
from binary.checking.two_sample_one_sided_sprt import two_sample_one_sided_sprt
from binary.checking.two_sample_two_sided_sprt import two_sample_two_sided_sprt


def one_sided_simulation_sprt(p_x, p_y, iter_size, batch_size, p0, d, alpha, beta, alternative):
    """
    Моделирование последовательного анализа
    параллельно в iter_size тестах
    для односторонней альтернативы

    :param p_x: реальное значение вероятности для первой вариации
    :param p_y: реальное значение вероятности для второй вариации
    :param iter_size: количество параллельных тестов в моделировании
    :param batch_size: размер одного батча генерирования данных и моделирования
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование односторонней альтернативы
    :return: словарь res
             res["duration"] - список длительностей теста
             res["result"] - список результатов теста
                             для alternative = "less"
                                -1: тест закончился, есть стат. значимое снижение вероятности
                                1: тест закончился, нет стат. значимого снижения вероятности
                             для alternative = "greater"
                                -1: тест закончился, нет стат. значимого повышения вероятности
                                1: тест закончился, есть стат. значимое повышение вероятности
             res["result_x_s"] - список значений S(n) на момент длительности теста для первой вариации
             res["result_y_s"] - список значений S(n) на момент длительности теста для второй вариации
    """
    # Суммарная длительность незаконченных тестов
    total_duration = 0

    # Характеристики законченных тестов
    completed_duration_list = []
    completed_result_list = []
    completed_x_s_list = []
    completed_y_s_list = []

    # Характеристики незаконченных тестов
    remain_iter_cnt = iter_size
    remain_last_curve = None
    remain_x_s_list = 0
    remain_y_s_list = 0

    # Итерируемся пока есть незаконченные тесты
    while remain_iter_cnt > 0:
        # Разыгрываем батч выборки для незаконченных тестов
        # и собираем статистику по этому батчу
        x = bernoulli.rvs(p_x, size=[remain_iter_cnt, batch_size])
        y = bernoulli.rvs(p_y, size=[remain_iter_cnt, batch_size])
        res = two_sample_one_sided_sprt(x, y, p0, d, alpha, beta,
                                        alternative=alternative,
                                        initial_curve=remain_last_curve)

        remain_duration_list = total_duration + res["duration"]
        remain_result_list = res["result"]
        last_curve = res["last_curve"]
        remain_x_s_list += res["result_x_s"]
        remain_y_s_list += res["result_y_s"]

        # Определяем незаконченные тесты
        remain_test_flg = remain_result_list == 0
        remain_iter_cnt = np.sum(remain_test_flg)

        # Рассчитываем характеристики законченных тестов
        completed_duration_list += list(remain_duration_list[~remain_test_flg])
        completed_result_list += list(remain_result_list[~remain_test_flg])
        completed_x_s_list += list(remain_x_s_list[~remain_test_flg])
        completed_y_s_list += list(remain_y_s_list[~remain_test_flg])

        # Рассчитываем характеристики незаконченных тестов
        remain_last_curve = last_curve[remain_test_flg]
        remain_x_s_list = remain_x_s_list[remain_test_flg]
        remain_y_s_list = remain_y_s_list[remain_test_flg]
        total_duration += batch_size

    return {
        "duration": np.array(completed_duration_list),
        "result": np.array(completed_result_list),
        "result_x_s": np.array(completed_x_s_list),
        "result_y_s": np.array(completed_y_s_list)
    }


def two_sided_simulation_sprt(p_x, p_y, iter_size, batch_size, p0, d, alpha, beta):
    """
    Моделирование последовательного анализа
    параллельно в iter_size тестах
    для двусторонней альтернативы

    :param p_x: реальное значение вероятности для первой вариации
    :param p_y: реальное значение вероятности для второй вариации
    :param iter_size: количество параллельных тестов в моделировании
    :param batch_size: размер одного батча генерирования данных и моделирования
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :return: словарь res
             res["duration"] - список длительностей теста
             res["result"] - список результатов теста
                                -1: тест закончился, нет стат. значимого изменения вероятности
                                1: тест закончился, есть стат. значимое изменение вероятности
             res["result_x_s"] - список значений S(n) на момент длительности теста для первой вариации
             res["result_y_s"] - список значений S(n) на момент длительности теста для второй вариации
    """
    # Суммарная длительность незаконченных тестов
    total_duration = 0

    # Характеристики законченных тестов
    completed_duration_list = []
    completed_result_list = []
    completed_x_s_list = []
    completed_y_s_list = []

    # Характеристики незаконченных тестов
    remain_iter_cnt = iter_size
    remain_greater_last_curve = None
    remain_less_last_curve = None
    remain_greater_stop_flg = None
    remain_less_stop_flg = None
    remain_x_s_list = 0
    remain_y_s_list = 0

    while remain_iter_cnt > 0:
        x = bernoulli.rvs(p_x, size=[remain_iter_cnt, batch_size])
        y = bernoulli.rvs(p_y, size=[remain_iter_cnt, batch_size])
        res = two_sample_two_sided_sprt(x, y, p0, d, alpha, beta,
                                        greater_initial_curve=remain_greater_last_curve,
                                        less_initial_curve=remain_less_last_curve,
                                        greater_stop_flg=remain_greater_stop_flg,
                                        less_stop_flg=remain_less_stop_flg)

        remain_duration_list = total_duration + res["duration"]
        remain_result_list = res["result"]
        remain_greater_last_curve = res["greater_last_curve"]
        remain_less_last_curve = res["less_last_curve"]
        remain_greater_stop_flg = res["greater_stop"]
        remain_less_stop_flg = res["less_stop"]
        remain_x_s_list += res["result_x_s"]
        remain_y_s_list += res["result_y_s"]

        remain_test_flg = remain_result_list == 0
        remain_iter_cnt = np.sum(remain_test_flg)

        completed_duration_list += list(remain_duration_list[~remain_test_flg])
        completed_result_list += list(remain_result_list[~remain_test_flg])
        completed_x_s_list += list(remain_x_s_list[~remain_test_flg])
        completed_y_s_list += list(remain_y_s_list[~remain_test_flg])

        remain_greater_last_curve = remain_greater_last_curve[remain_test_flg]
        remain_less_last_curve = remain_less_last_curve[remain_test_flg]
        remain_x_s_list = remain_x_s_list[remain_test_flg]
        remain_y_s_list = remain_y_s_list[remain_test_flg]
        total_duration += batch_size
        remain_greater_stop_flg = remain_greater_stop_flg[remain_test_flg]
        remain_less_stop_flg = remain_less_stop_flg[remain_test_flg]

    return {
        "duration": np.array(completed_duration_list),
        "result": np.array(completed_result_list),
        "result_x_s": np.array(completed_x_s_list),
        "result_y_s": np.array(completed_y_s_list)
    }


def simulation_sprt(p_x, p_y, iter_size, batch_size, p0, d, alpha, beta, alternative):
    """
    Моделирование последовательного анализа
    параллельно в iter_size тестах

    :param p_x: реальное значение вероятности для первой вариации
    :param p_y: реальное значение вероятности для второй вариации
    :param iter_size: количество параллельных тестов в моделировании
    :param batch_size: размер одного батча генерирования данных и моделирования
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование альтернативы
    :return: словарь res
             res["duration"] - список длительностей теста
             res["result"] - список результатов теста
                             для alternative = "less"
                                -1: тест закончился, есть стат. значимое снижение вероятности
                                1: тест закончился, нет стат. значимого снижения вероятности
                             для alternative = "greater"
                                -1: тест закончился, нет стат. значимого повышения вероятности
                                1: тест закончился, есть стат. значимое повышение вероятности
                             для alternative = "two-sided"
                                -1: тест закончился, нет стат. значимого изменения вероятности
                                1: тест закончился, есть стат. значимое изменение вероятности
             res["result_s"] - список значений S(n) на момент длительности теста
    """
    if alternative == "two-sided":
        return two_sided_simulation_sprt(p_x, p_y, iter_size, batch_size, p0, d, alpha, beta)
    else:
        return one_sided_simulation_sprt(p_x, p_y, iter_size, batch_size, p0, d, alpha, beta, alternative)
