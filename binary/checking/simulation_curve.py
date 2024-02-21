import numpy as np


def one_sample_curve(s_list, p0, d, alpha, beta, alternative):
    """
    Определение кривой логарифмического отношения правдоподобий

    :param s_list: массив размера [iter_size, sample_size],
                   где каждая строка - накопленная сумма S(n) данных из {0, 1}
                   для каждого n <= sample_size,
                   а iter_size - количество итераций моделирования
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование односторонней альтернативы
    :return: словарь res
             res["curve"] - кривая логарифмического отношения правдоподобий
             res["low_bound"] - нижняя граница для логарифмического отношения правдоподобий
             res["high_bound"] - верхняя граница для логарифмического отношения правдоподобий
    """

    # Количество итераций моделирования и длительность получения данных
    iter_size = s_list.shape[0]
    sample_size = s_list.shape[1]

    # Определение параметров последовательного теста
    if alternative == "greater":
        p_low = p0
        p_high = p0 + d
        alpha_low = beta
        alpha_high = alpha
    elif alternative == "less":
        p_low = p0 - d
        p_high = p0
        alpha_low = alpha
        alpha_high = beta
    else:
        raise ValueError(f"Неправильная альтернатива: {alternative}")

    # Получение массива из строк вида (1, ..., sample_size) в количестве iter_size
    n_one_dim_list = 1 + np.arange(sample_size)
    n_list = np.tile(n_one_dim_list.reshape(-1, 1), iter_size).T

    # Логарифмическое отношение правдоподобий для бернуллиевских величин
    curve = s_list * np.log(p_high / p_low) \
            + (n_list - s_list) * np.log((1 - p_high) / (1 - p_low))

    # Определение порогов для логарифмического отношения правдоподобий
    low_bound = np.log(alpha_low / (1 - alpha_high))
    high_bound = np.log((1 - alpha_low) / alpha_high)

    return {
        "curve": curve,
        "low_bound": low_bound,
        "high_bound": high_bound
    }
