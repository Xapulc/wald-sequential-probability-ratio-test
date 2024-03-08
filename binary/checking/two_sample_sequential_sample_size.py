from binary.checking.tools import transform_two_sample_one_sided_mde
from binary.checking.one_sample_sequential_sample_size import sequential_sample_size as one_sample_sequential_sample_size, \
                                                              max_sequential_sample_size as one_sample_max_sequential_sample_size


def sequential_sample_size(p, p0, d, alpha, beta, alternative, first_prop=0.5):
    """
    Расчёт средней длительности
    последовательного двухвыборочного теста

    :param p: истинное значение вероятности,
              лежащее отличающееся от p0 не более чем на d
    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование альтернативы,
                        пока что двусторонняя альтернатива не поддерживается
    :param first_prop: доля первой выборки, пока что = 0.5
    :return: [размер первой выборки, размер второй выборки]
    """
    p0_transformed = 1 / 2

    if alternative in ("less", "greater"):
        d_transformed = transform_two_sample_one_sided_mde(p0, d, alternative=alternative)
        diff_transformed = transform_two_sample_one_sided_mde(p0, p-p0, alternative=alternative)\

        if alternative == "greater":
            p_transformed = p0_transformed + diff_transformed
        else:
            p_transformed = p0_transformed - diff_transformed

        sample_size = one_sample_sequential_sample_size(p_transformed, p0_transformed, d_transformed, alpha, beta, alternative)

        if alternative == "greater":
            scale = p0 * (1 - (p0 + d)) + (p0 + d) * (1 - p0)
        else:
            scale = p0 * (1 - (p0 - d)) + (p0 - d) * (1 - p0)

        sample_size /= scale
        return sample_size, sample_size
    else:
        raise ValueError(f"Неправильная альтернатива: {alternative}")


def max_sequential_sample_size(p0, d, alpha, beta, alternative, first_prop=0.5):
    """
    Расчёт максимальной средней длительности
    последовательного двухвыборочного теста

    :param p0: значение вероятности при гипотезе
    :param d: абсолютное значение MDE
    :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
    :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
    :param alternative: наименование альтернативы,
                        пока что двусторонняя альтернатива не поддерживается
    :param first_prop: доля первой выборки, пока что = 0.5
    :return: [размер первой выборки, размер второй выборки]
    """
    p0_transformed = 1 / 2

    if alternative in ("less", "greater"):
        d_transformed = transform_two_sample_one_sided_mde(p0, d, alternative=alternative)
        max_sample_size, max_p = one_sample_max_sequential_sample_size(p0_transformed, d_transformed, alpha, beta, alternative)

        if alternative == "greater":
            scale = p0 * (1 - (p0 + d)) + (p0 + d) * (1 - p0)
        else:
            scale = p0 * (1 - (p0 - d)) + (p0 - d) * (1 - p0)

        max_sample_size /= scale
        return [max_sample_size, max_sample_size]
    else:
        raise ValueError(f"Неправильная альтернатива: {alternative}")
