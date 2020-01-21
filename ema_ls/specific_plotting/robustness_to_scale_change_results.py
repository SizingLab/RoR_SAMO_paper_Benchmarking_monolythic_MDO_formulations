import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rcParams['text.latex.unicode'] = True


# matplotlib.rcParams.update({'font.size': 14})


def robustness_to_scale_change_results():
    # MDF formulation and SLSQP optimizer with full_analytic
    k_os = [1.23, 1.26, 1.31, 1.35]

    # full analytic
    # mdf = np.array([2858, 0, 0, 0])
    # idf = np.array([780, 969, 894, 1721])
    # hybrid = np.array([1229, 1465, 2049, 69436])
    # nvh = np.array([895, 1035, 1297, 2140])

    # semi analytic fd
    mdf = np.array([13833, 0, 0, 0])
    idf = np.array([2305, 2979, 2969, 7491])
    hybrid = np.array([3714, 4450, 5794, 116426])
    nvh = np.array([2207, 2735, 3709, 5572])

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    N = 4
    ind = np.array([1, 2, 3, 4])

    # ind = np.arange(N)    # the x locations for the groups
    width = 0.15  # the width of the bars: can also be len(x) sequence

    ax.bar(ind - width * 3 / 2, mdf, width, label='MDF', align='center')
    ax.bar(ind - width / 2, idf, width, label='IDF', align='center')
    ax.bar(ind + width / 2, hybrid, width, label='HYBRID', align='center')
    ax.bar(ind + width * 3 / 2, nvh, width, label='NVH', align='center')

    ax.set_yscale('log')
    plt.gca().yaxis.grid(True, linestyle=':')

    plt.xticks(ind, (r'$k_{F_{ema}} = 1$' + '\n' + r'$k_{os} = ' + str(k_os[0]) + '$', \
                     r'$k_{F_{ema}} = 2$' + '\n' + r'$k_{os} = ' + str(k_os[1]) + '$', \
                     r'$k_{F_{ema}} = 5$' + '\n' + r'$k_{os} = ' + str(k_os[2]) + '$', \
                     r'$k_{F_{ema}} = 10$' + '\n' + r'$k_{os} = ' + str(k_os[3]) + '$'))

    plt.legend()
    ax.set_ylabel('Number of function evaluations [-]')

    plt.show()


if __name__ == "__main__":
    robustness_to_scale_change_results()
