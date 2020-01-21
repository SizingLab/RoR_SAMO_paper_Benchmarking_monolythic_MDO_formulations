import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rcParams['text.latex.unicode'] = True


# matplotlib.rcParams.update({'font.size': 14})


def robustness_to_scale_change_results():
    # MDF formulation and SLSQP optimizer with full_analytic
    k_os = [1.22, 1.42, 1.53, 1.6]

    # full analytic
    mdf = np.array([24, 14, 51, 6525])
    idf = np.array([14, 0, 0, 0])
    hybrid = np.array([9, 5, 9, 6])
    nvh = np.array([8, 5, 7, 6])

    # semi analytic fd
    # mdf = np.array([139, 79, 221, 6238])
    # idf = np.array([84, 0, 0, 0])
    # hybrid = np.array([49, 30, 54, 36])
    # nvh = np.array([36, 25, 31, 26])

    # monolythic fd
    # mdf = np.array([757, 295, 553, 0])
    # idf = np.array([168, 0, 0, 0])
    # hybrid = np.array([88, 55, 99, 66])
    # nvh = np.array([78, 55, 67, 58])

    # cobyla
    # mdf = np.array([186, 295, 0, 0])
    # idf = np.array([0, 0, 0, 0])
    # hybrid =  np.array([587, 0, 0, 0])
    # nvh =  np.array([26, 25, 29, 87])

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
                     r'$k_{F_{ema}} = 10$' + '\n' + r'$k_{os} = ' + str(k_os[1]) + '$', \
                     r'$k_{F_{ema}} = 20$' + '\n' + r'$k_{os} = ' + str(k_os[2]) + '$', \
                     r'$k_{F_{ema}} = 30$' + '\n' + r'$k_{os} = ' + str(k_os[3]) + '$'))

    plt.legend()
    ax.set_ylabel('Number of function evaluations [-]')

    plt.show()


if __name__ == "__main__":
    robustness_to_scale_change_results()
