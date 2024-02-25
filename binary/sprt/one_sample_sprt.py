import numpy as np


class BinaryOneSampleSprt(object):
    def __init__(self, p0, d, alpha=0.05, beta=0.2, alternative="two-sided",
                 initial_success_cnt=0, initial_sample_size=0):
        """
        Последовательный анализ в случае одновыборочной задачи

        :param p0: значение вероятности при гипотезе
        :param d: абсолютное значение MDE
        :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
        :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
        :param alternative: наименование односторонней альтернативы
                            greater: правосторонняя альтернатива p > p0
                            less: левосторонняя альтернатива p < p0
                            two-sided: двусторонняя альтернатива p != p0
        :param initial_success_cnt: изначальное количество "успехов"
        :param initial_sample_size: изначальный размер выборки
        """
        # Параметры последовательного теста
        self.p0 = p0
        self.d = np.abs(d)
        self.alpha = alpha
        self.beta = beta
        self.alternative = alternative

        # Параметры текущего состояния теста
        self.success_cnt = initial_success_cnt
        self.sample_size = initial_sample_size

        # Принятие решения
        self.stop_success_cnt = self.success_cnt
        self.stop_sample_size = self.sample_size
        self.decision_desc = "Тест продолжается"

        # Признак остановки последовательного анализа
        # для двусторонней альтернативы
        self.greater_stop_flg = False
        self.less_stop_flg = False

    def calc_one_sided_probs(self, alternative):
        """
        Функция для расчёта базовых значений вероятностей (конверсий)
        (нижней и верхней)

        :param alternative: наименование односторонней альтернативы
                            greater: правосторонняя альтернатива p > p0
                            less: левосторонняя альтернатива p < p0
        :return: нижнее значение вероятности, верхнее значение вероятности
        """

        if alternative == "greater":
            p_low = self.p0
            p_high = self.p0 + self.d
        elif alternative == "less":
            p_low = self.p0 - self.d
            p_high = self.p0
        else:
            raise ValueError(f"Неправильная альтернатива: {alternative}")

        return p_low, p_high

    def calc_one_sided_bounds(self, alpha, beta, alternative):
        """
        Функция для односторонней альтернативы
        рассчитывает пороговые значения,
        при пересечении которых тест останавливается и принимается решение

        :param alpha: ограничение на вероятность ошибки I рода (уровень значимости)
        :param beta: ограничение на вероятность ошибки II рода (1 - мощность)
        :param alternative: наименование односторонней альтернативы
                            greater: правосторонняя альтернатива p > p0
                            less: левосторонняя альтернатива p < p0
        :return: нижняя граница, верхняя граница
        """

        if alternative == "greater":
            low_bound = np.log(beta / (1 - alpha))
            high_bound = np.log((1 - beta) / alpha)
        elif alternative == "less":
            low_bound = np.log(alpha / (1 - beta))
            high_bound = np.log((1 - alpha) / beta)
        else:
            raise ValueError(f"Неправильная альтернатива: {alternative}")

        return low_bound, high_bound

    def calc_one_sided_curve(self, success_cnt, sample_size, alternative):
        """
        Функция для расчёта значений логарифмического отношения правдоподобий

        :param success_cnt: количество "успехов"
        :param sample_size: размер выборки
        :param alternative: наименование односторонней альтернативы
                            greater: правосторонняя альтернатива p > p0
                            less: левосторонняя альтернатива p < p0
        :return:
        """
        p_low, p_high = self.calc_one_sided_probs(alternative)

        # Логарифмическое отношение правдоподобий для бернуллиевских величин
        curve = success_cnt * np.log(p_high / p_low) \
                + (sample_size - success_cnt) * np.log((1 - p_high) / (1 - p_low))

        return curve

    def append(self, x):
        """
        Добавление нового элемента выборки
        с принятием решения о возможности
        остановки последовательного теста

        :param x: значение нового элемента выборки
        :return: описание принятого решения
        """

        # Обновление общей статистики теста
        self.success_cnt += x
        self.sample_size += 1

        # Если тест продолжается, обновляем расчёты
        if self.decision_desc == "Тест продолжается":
            # Так как тест ещё не остановлен,
            # обновляем статистику теста до принятого решения
            self.stop_success_cnt = self.success_cnt
            self.stop_sample_size = self.sample_size

            if self.alternative != "two-sided":
                # Если альтернатива одностороняя,
                # то решение принимается по одному расчёту
                # логарифмического отношения правдоподобий
                curve = self.calc_one_sided_curve(self.success_cnt, self.sample_size, self.alternative)
                low_bound, high_bound = self.calc_one_sided_bounds(self.alpha, self.beta, self.alternative)

                # Если значение логарифмического отношения правдоподобий
                # пересекает одну из границ,
                # тест останавливается с принятием решения
                if curve > high_bound:
                    if self.alternative == "greater":
                        self.decision_desc = "Тест остановлен, справедлива альтернатива p > p0"
                    else:
                        self.decision_desc = "Тест остановлен, справедлива гипотеза p >= p0"
                elif curve < low_bound:
                    if self.alternative == "greater":
                        self.decision_desc = "Тест остановлен, справедлива гипотеза p <= p0"
                    else:
                        self.decision_desc = "Тест остановлен, справедлива альтернатива p < p0"
            else:
                # Если альтернатива двусторонняя,
                # то мы параллельно "проводим" два последовательных анализа:
                # p0 против p0+d (alternative = "greater"),
                # p0-d против p0 (alternative = "less")
                greater_curve = self.calc_one_sided_curve(self.success_cnt, self.sample_size, alternative="greater")
                greater_low_bound, greater_high_bound = self.calc_one_sided_bounds(self.alpha/2, self.beta, alternative="greater")

                less_curve = self.calc_one_sided_curve(self.success_cnt, self.sample_size, alternative="less")
                less_low_bound, less_high_bound = self.calc_one_sided_bounds(self.alpha/2, self.beta, alternative="less")

                # Если тест для alternative = "greater" ранее не завершён,
                # а сейчас произошло пересечение верхней границы,
                # то останавливаем тест с решением о стат. значимом росте
                if not self.greater_stop_flg and greater_curve > greater_high_bound:
                    self.decision_desc = "Тест остановлен, справедлива альтернатива p > p0"

                # Если тест для alternative = "less" ранее не завершён,
                # а сейчас произошло пересечение нижней границы,
                # то останавливаем тест с решением о стат. значимом падении
                if not self.less_stop_flg and less_curve < less_low_bound:
                    self.decision_desc = "Тест остановлен, справедлива альтернатива p < p0"

                # Если для какой-то из альтернатив тест был ранее завершён,
                # но тест с двусторонней альтернативой продолжается,
                # то ранее было пересечение границы, соответствующее p = p0
                # Поэтому если для какой-то альтернативы тест завершился ранее,
                # а сейчас для другой альтернативы
                # есть пересечение границы, соответствующее  p = p0,
                # то мы можем завершить тест с принятием решения p = p0
                if self.greater_stop_flg and less_curve > less_high_bound:
                    self.decision_desc = "Тест остановлен, справедлива гипотеза p = p0"
                if self.less_stop_flg and greater_curve < greater_low_bound:
                    self.decision_desc = "Тест остановлен, справедлива гипотеза p = p0"

                # Если ни для какой альтернативы тест ранее не был завершён,
                # а сейчас для обеих альтернатив есть пересечение границы при p = p0,
                # то мы можем завершить тест с принятием решения p = p0
                if not self.greater_stop_flg and greater_curve < greater_low_bound \
                    and self.less_stop_flg and less_curve > less_high_bound:
                    self.decision_desc = "Тест остановлен, справедлива гипотеза p = p0"

                # Завершаем тест для тех альтернатив,
                # для которых есть пересечение хотя бы одной из границ
                if greater_curve > greater_high_bound or greater_curve < greater_low_bound:
                    self.greater_stop_flg = True
                if less_curve > less_high_bound or less_curve < less_low_bound:
                    self.less_stop_flg = True

        return self.decision_desc
