import numpy as np
from scipy.stats import binomtest, ttest_1samp


def get_duration_from_bound_crossing(bound_crossing_flg):
    """
    Функция для расчёта длительности теста
    по массиву индикаторов пересечения границы

    :param bound_crossing_flg: массив размера [iter_size, sample_size],
                               где каждая строка - bool значение
                               справедливости события пересечения границы,
                               а iter_size - количество итераций моделирования (тестов)
    :return: список размера iter_size из длительностей теста
    """
    max_sample_size = bound_crossing_flg.shape[1]
    first_bound_crossing_index_list = np.where(bound_crossing_flg.any(axis=1) > 0,
                                               np.argmax(bound_crossing_flg, axis=1),
                                               max_sample_size-1)
    return 1 + first_bound_crossing_index_list


def get_value_at_duration(value_list, duration_list):
    """
    Функция для расчёта значения величин
    на момент длительности теста

    :param value_list: массив размера [iter_size, sample_size],
                       где каждая строка - значения какой-то величины на тесте,
                       а iter_size - количество итераций моделирования (тестов)
    :param duration_list: список размера iter_size из длительностей теста
    :return: список размера iter_size из значений величин
    """
    duration_index_list = duration_list.reshape(-1, 1) - 1
    result_list = np.take_along_axis(value_list,
                                     duration_index_list,
                                     axis=1)
    return result_list[:, 0]


def freq_conf_interval(freq_list, sample_size, conf=0.99):
    """
    Функция для построения статистически незначимых
    отклонений от частоты события

    :param freq_list: список частот
    :param sample_size: знаменатель частот
    :param conf: уровень доверия
    :return: список из нижних и верхних
             статистически незначимых отклонений
    """
    left_side_list = []
    right_side_list = []

    for freq in freq_list:
        res = binomtest(int(freq * sample_size), sample_size)
        left_side, right_side = res.proportion_ci(conf)
        left_side_list.append(freq - left_side)
        right_side_list.append(right_side - freq)

    return left_side_list, right_side_list


def duration_conf_interval(duration_matrix, conf=0.99):
    """
    Функция для построения статистически незначимых
    отклонений от средней длительности теста

    :param duration_matrix: список из duration_list - списков значений длительностей теста
    :param conf: уровень доверия
    :return: список из нижних и верхних
             статистически незначимых отклонений
    """
    left_side_list = []
    right_side_list = []

    for duration_list in duration_matrix:
        mean = np.mean(duration_list)
        res = ttest_1samp(duration_list, popmean=0)
        left_side, right_side = res.confidence_interval(conf)
        left_side_list.append(mean - left_side)
        right_side_list.append(right_side - mean)

    return left_side_list, right_side_list
