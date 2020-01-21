from ema.core.benchmark_ema import BenchmarkEMA

from ema.core.benchmark_ema import BenchmarkEMA as BenchmarkEMALS

from sellar.core.benchmark_sellar import BenchmarkSellar

# Set to True for running the benchmark of the high dynamic EMA test problem
RUN_EMA_BENCHMARK = True

# Set to True for running the benchmark of the larger scale high dynamic EMA test problem
RUN_EMA_LS_BENCHMARK = True

# Set to True for running the benchmark of the Sellar test problem
RUN_SELLAR_BENCHMARK = True


if __name__ == '__main__':

    if RUN_EMA_BENCHMARK:
        bm = BenchmarkEMA()
        bm.run()

    if RUN_EMA_LS_BENCHMARK:
        bm = BenchmarkEMALS()
        bm.run()

    if RUN_SELLAR_BENCHMARK:
        bm = BenchmarkSellar()
        bm.run()
