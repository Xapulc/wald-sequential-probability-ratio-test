import numpy as np
from scipy.stats import norm


def one_sided_classic_sample_size(p_low, p_high, alpha_low, alpha_high):
    """
    Расчёт размера выборки в классическом дизайне
    одновыборочного теста с фиксированной длительностью теста
    и односторонней альтернативой

    :param p_low: нижний порог веротяности
    :param p_high: верхний порог вероятности
    :param alpha_low: вероятность ошибки для нижнего порога
    :param alpha_high: вероятность ошибки для верхнего порога
    :return: размер выборки
    """

    assert 0 <= p_low and p_low <= 1, f"Вероятность {p_low:%} вне [0, 1]"
    assert 0 <= p_high and p_high <= 1, f"Вероятность {p_high:%} вне [0, 1]"

    d = p_high - p_low
    summand_low = norm.ppf(1-alpha_low) * np.sqrt(p_low * (1-p_low))
    summand_high = norm.ppf(1-alpha_high) * np.sqrt(p_high * (1-p_high))

    return int(np.ceil(((summand_low + summand_high)**2) / (d**2)))


def classic_sample_size(p0, d, alpha, beta, alternative):
    """
    Расчёт размера выборки в классическом дизайне
    одновыборочного теста с фиксированной длительностью теста

    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование альтернативы
    :return: размер выборки
    """

    if alternative == "less":
        p_low = p0 - np.abs(d)
        p_high = p0
        return one_sided_classic_sample_size(p_low, p_high, beta, alpha)
    elif alternative == "greater":
        p_low = p0
        p_high = p0 + np.abs(d)
        return one_sided_classic_sample_size(p_low, p_high, alpha, beta)
    elif alternative == "two-sided":
        return max(one_sided_classic_sample_size(p0-np.abs(d), p0, beta, alpha/2),
                   one_sided_classic_sample_size(p0, p0+np.abs(d), alpha/2, beta))
    else:
        raise ValueError(f"Неправильная альтернатива: {alternative}")
