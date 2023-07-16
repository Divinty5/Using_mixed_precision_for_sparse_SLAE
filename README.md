В рамках проекта "Использование смешанной точности для разреженных СЛАУ" предполагается реализация метода сопряжённых градиентов ([CG](https://ru.m.wikipedia.org/wiki/Метод_сопряжённых_градиентов_(для_решения_СЛАУ))) с предобуславливанием [ILU(0)](https://en.m.wikipedia.org/wiki/Incomplete_LU_factorization) и [ICF](https://en.m.wikipedia.org/wiki/Incomplete_Cholesky_factorization).

Также проводится исследование с анализом работы данных алгоритмов, в результате которого можно будет выявить места, где можно будет "сэкономить" память, выделяемую для переменных, путём понижения точности от [double](https://ru.m.wikipedia.org/wiki/Число_двойной_точности) до [single](https://ru.m.wikipedia.org/wiki/Число_одинарной_точности) precision. Планируется выбор оптимального решения конкретных матриц в смысле времени работы метода, а также суммарного числа итераций при сохранении достаточно хорошего приближения (по условию на норму невязки).
