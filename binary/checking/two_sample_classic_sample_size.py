import numpy as np
from scipy.stats import norm


def one_sided_classic_sample_size(p_low, p_high, alpha_low, alpha_high, first_prop=0.5):
    """
    Расчёт размера выборки в классическом дизайне
    двухвыборочного теста с фиксированной длительностью теста
    и односторонней альтернативой

    :param p_low: нижний порог веротяности
    :param p_high: верхний порог вероятности
    :param alpha_low: вероятность ошибки для нижнего порога
    :param alpha_high: вероятность ошибки для верхнего порога
    :param first_prop: доля первой выборки, пока что = 0.5
    :return: размер выборки
    """

    assert 0 <= p_low and p_low <= 1, f"Вероятность {p_low:%} вне [0, 1]"
    assert 0 <= p_high and p_high <= 1, f"Вероятность {p_high:%} вне [0, 1]"

    quantile_factor = norm.ppf(1-alpha_low) + norm.ppf(1-alpha_high)
    effect_size = 2 * (np.arcsin(np.sqrt(p_high)) - np.arcsin(np.sqrt(p_low)))

    return int(np.ceil(2 * (quantile_factor**2) / (effect_size**2)))


def classic_sample_size(p0, d, alpha, beta, alternative, first_prop=0.5):
    """
    Расчёт размера выборки в классическом дизайне
    двухвыборочного теста с фиксированной длительностью теста

    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование альтернативы,
                        пока что двусторонняя альтернатива не поддерживается
    :param first_prop: доля первой выборки, пока что = 0.5
    :return: [размер первой выборки, размер второй выборки]
    """

    if alternative == "less":
        p_low = p0 - np.abs(d)
        p_high = p0
        sample_size = one_sided_classic_sample_size(p_low, p_high, alpha, beta)
        return [sample_size, sample_size]
    elif alternative == "greater":
        p_low = p0
        p_high = p0 + np.abs(d)
        sample_size = one_sided_classic_sample_size(p_low, p_high, alpha, beta)
        return [sample_size, sample_size]
    elif alternative == "two-sided":
        sample_size = max(one_sided_classic_sample_size(p0-np.abs(d), p0, alpha/2, beta),
                          one_sided_classic_sample_size(p0, p0+np.abs(d), alpha/2, beta))
        return [sample_size, sample_size]
    else:
        raise ValueError(f"Неправильная альтернатива: {alternative}")
