import numpy as np
from scipy.optimize import bisect, minimize_scalar


def p_critical(p_low, p_high):
    """
    Определение критического значения, при котором равно нулю
    математическое ожидание логарифмического отношения правдоподобий
    при проверке гипотезы p = p_low против альтернативы p = p_high

    :param p_low: нижний порог веротяности
    :param p_high: верхний порог вероятности
    :return: p - критическое значение
    """
    return np.log((1-p_low) / (1-p_high)) / (np.log(p_high / p_low) + np.log((1-p_low) / (1-p_high)))


def operation_characteristic_root(p, p_low, p_high):
    """
    Значение параметра для определения значения
    операционной характеристики
    - вероятности принять гипотезу p = p_low
    при истинное значение равно p

    Вальд А.
    Последовательный анализ.
    – 1960. – С. 75-79.

    :param p: истинное значение вероятности,
              лежащее в отрезке [p_low, p_high]
    :param p_low: нижний порог веротяности
    :param p_high: верхний порог вероятности
    :return: значение параметра, лежит в [-1, 1]
    """
    def helper(h):
        """
        Функция, корень которой является
        искомым параметром h для определения
        значения операционной характеристики

        :param h: искомый параметр
        :return:
        """
        return -1 + p * ((p_high / p_low)**h) \
               + (1-p) * (((1-p_low) / (1-p_high))**(-h))

    # Значение параметра h, при котором
    # производная функции helper(h) равна нулю
    h_crit = np.log(((1-p) * np.log((1-p_low) / (1-p_high))) / (p * np.log(p_high / p_low))) \
             / (np.log((1-p_low) / (1-p_high)) + np.log(p_high / p_low))

    # Критическое значение параметра p
    p_crit = p_critical(p_low, p_high)

    if p == p_low:
        # При p = p_low можно вывести аналитически, что h = 1
        return 1
    elif p > p_low and p < p_crit:
        # Численное нахождение корня на отрезке [h_crit, 1]
        # Функция на этом отрезке монотонна
        return bisect(helper, h_crit, 1, xtol=(p_crit-p) / 10_000)
    elif p == p_crit:
        # При p = p_crit можно вывести аналитически, что h = 0
        return 0
    elif p > p_crit and p < p_high:
        # Численное нахождение корня на отрезке [-1, h_crit]
        # Функция на этом отрезке монотонна
        return bisect(helper, -1, h_crit, xtol=(p-p_crit) / 10_000)
    elif p == p_high:
        # При p = p_high можно вывести аналитически, что h = -1
        return -1
    else:
        raise ValueError(f"Величина {p} должна находиться "
                         f"в отрезке [{p_low}, {p_high}]")


def operation_characteristic(p, p_low, p_high, alpha_low, alpha_high):
    """
    Операционная характеристика
    - вероятность принять гипотезу p = p_low
    при истинное значение равно p

    Tartakovsky A., Nikiforov I., Basseville M.
    Sequential analysis: Hypothesis testing and changepoint detection.
    – CRC press, 2014. – С. 125.

    :param p: истинное значение вероятности,
              лежащее в отрезке [p_low, p_high]
    :param p_low: нижний порог веротяности
    :param p_high: верхний порог вероятности
    :param alpha_low: вероятность пересечения нижней границы
                      при справедливости p = p_high
    :param alpha_high: вероятность пересечения нижней границы
                       при справедливости p = p_low
    :return: значение операционной характеристики
    """
    # Границы принятия решений
    low_bound = np.log(alpha_low / (1 - alpha_high))
    high_bound = np.log((1 - alpha_low) / alpha_high)

    # Критическое значение параметра p
    p_crit = p_critical(p_low, p_high)

    if p == p_crit:
        # Если значение параметра p равно критическому,
        # можно аналитически определить значени операционной характеристики
        return high_bound / (high_bound - low_bound)
    else:
        # Иначе нужно определить значения параметра операционной характеристики,
        # и после воспользоваться аналитической формулой
        h = operation_characteristic_root(p, p_low, p_high)
        return (np.exp(high_bound*h) - 1) / (np.exp(high_bound*h) - np.exp(low_bound*h))


def one_sided_sequential_sample_size(p, p_low, p_high, alpha_low, alpha_high):
    """
    Расчёт средней длительности
    последовательного одновыборочного теста
    при односторонней альтернативе

    Tartakovsky A., Nikiforov I., Basseville M.
    Sequential analysis: Hypothesis testing and changepoint detection.
    – CRC press, 2014. – С. 125.

    :param p: истинное значение вероятности,
              лежащее в отрезке [p_low, p_high]
    :param p_low: нижний порог веротяности
    :param p_high: верхний порог вероятности
    :param alpha_low: вероятность пересечения нижней границы
                      при справедливости p = p_high
    :param alpha_high: вероятность пересечения нижней границы
                       при справедливости p = p_low
    :return: средняя длительность теста
    """
    # Границы принятия решений
    low_bound = np.log(alpha_low / (1 - alpha_high))
    high_bound = np.log((1 - alpha_low) / alpha_high)

    # Критическое значение параметра p
    p_crit = p_critical(p_low, p_high)
    # Значение операционной характеристики
    o_c = operation_characteristic(p, p_low, p_high, alpha_low, alpha_high)

    if np.abs(p - p_crit) < min(1 / 10_000, p / 100, (1 - p) / 100):
        # Если значение p является критическим,
        # используем квадратичную формулу
        e_z_sqr = p * (np.log(p_high / p_low)**2) + (1 - p) * (np.log((1-p_low) / (1-p_high))**2)
        return int(np.ceil((o_c * (low_bound**2) + (1-o_c) * (high_bound**2)) / e_z_sqr))
    else:
        # Иначе используем обычную формулу
        e_z = p * np.log(p_high / p_low) - (1 - p) * np.log((1-p_low) / (1-p_high))
        return int(np.ceil((o_c * low_bound + (1-o_c) * high_bound) / e_z))


def sequential_sample_size(p, p0, d, alpha, beta, alternative):
    """
    Расчёт средней длительности
    последовательного одновыборочного теста

    :param p: истинное значение вероятности,
              лежащее отличающееся от p0 не более чем на d
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование альтернативы
    :return: средняя длительность теста
    """
    if alternative == "less":
        p_low = p0 - np.abs(d)
        p_high = p0
        return one_sided_sequential_sample_size(p, p_low, p_high, alpha, beta)
    elif alternative == "greater":
        p_low = p0
        p_high = p0 + np.abs(d)
        return one_sided_sequential_sample_size(p, p_low, p_high, beta, alpha)
    else:
        raise ValueError(f"Неправильная альтернатива: {alternative}")


def max_one_sided_sequential_sample_size(p_low, p_high, alpha_low, alpha_high):
    """
    Расчёт максимального значения средней длительности
    последовательного одновыборочного теста
    при односторонней альтернативе

    Tartakovsky A., Nikiforov I., Basseville M.
    Sequential analysis: Hypothesis testing and changepoint detection.
    – CRC press, 2014. – С. 125.

    :param p_low: нижний порог веротяности
    :param p_high: верхний порог вероятности
    :param alpha_low: вероятность пересечения нижней границы
                      при справедливости p = p_high
    :param alpha_high: вероятность пересечения нижней границы
                       при справедливости p = p_low
    :return: [средняя длительность теста, значение p]
    """
    low_bound = np.log(alpha_low / (1 - alpha_high))
    high_bound = np.log((1 - alpha_low) / alpha_high)

    def p(h):
        if np.abs(h) < 1e-8:
            return p_critical(p_low, p_high)
        else:
            return (1 - np.exp(np.log((1-p_high) / (1-p_low)) * h)) \
                   / (np.exp(np.log(p_high / p_low) * h) - np.exp(np.log((1-p_high) / (1-p_low)) * h))

    def f(a, b, h):
        return (np.exp(a*h) - np.exp(b*h)) / (a * (1 - np.exp(b*h)) + b * (np.exp(a*h) - 1))

    def helper(h):
        if np.abs(h) < 1e-8:
            return -low_bound * high_bound / (np.log(p_high / p_low) * np.log((1-p_high) / (1-p_low)))
        else:
            return -f(np.log(p_high / p_low), np.log((1-p_high) / (1-p_low)), h) \
                   / f(high_bound, low_bound, h)

    res = minimize_scalar(helper, bounds=[-1, 1])
    return int(np.ceil(-res.fun)), p(res.x)


def max_sequential_sample_size(p0, d, alpha, beta, alternative):
    """
    Расчёт максимального значения средней длительности
    последовательного одновыборочного теста

    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование альтернативы
    :return: [средняя длительность теста, значение p]
    """
    if alternative == "less":
        p_low = p0 - np.abs(d)
        p_high = p0
        return max_one_sided_sequential_sample_size(p_low, p_high, alpha, beta)
    elif alternative == "greater":
        p_low = p0
        p_high = p0 + np.abs(d)
        return max_one_sided_sequential_sample_size(p_low, p_high, beta, alpha)
    else:
        raise ValueError(f"Неправильная альтернатива: {alternative}")
