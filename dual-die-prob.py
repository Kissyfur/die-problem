import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import os

import time
from progress.bar import IncrementalBar as Bar
import measures as m
import algorithms
import grapher as g

sgd = algorithms.SGD()
sngd = algorithms.SNGD()
megd = algorithms.MEGD()
csngd = algorithms.CSNGD()
csngd2 = algorithms.CSNGD2()
dsngd = algorithms.DSNGD()
mod1000 = algorithms.MOD(1000)
mod1 = algorithms.MOD(1)
map = algorithms.MAP()

def generate_sample(beta, n):
    return np.random.choice(len(beta), n, p=algorithms.to_p(beta))


def run_algorithm(alg, beta, sample, real_beta, file=None):
    estimations = None
    try:
        estimations = np.load(file + '.npy')
    except:
        print("Estimations not found in", file, ". Running the experiment")
        hyperparams = alg.adjust_hyperparams(beta, sample, real_beta)
        try:
            estimations = alg.learn(beta, hyperparams, sample)
            if file:
                try:
                    np.save(file, estimations)
                except:
                    pass
        except (OverflowError, np.linalg.linalg.LinAlgError, FloatingPointError):
            print("The best Learning rate failed to converge in whole sample")
            exit()
    return estimations


def experiment(n_classes, train_size, algs, many_experiments=1, extremality=1, save=True):
    m_vals = {}
    for alg in algs:
        m_vals[alg.name] = []
    current_path = os.getcwd()
    # os.path.expanduser('~/')
    directory = '/saved_data/'
    path = current_path + directory

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    for i in range(many_experiments):
        np.random.seed(i)
        beta = np.random.normal(loc=0, scale=extremality, size=n_classes)
        beta[0] = 0
        entropy = m.entropy(beta)
        max_entropy = m.max_entropy(n_classes)
        print(i, "Real beta has dim:", n_classes, "and extremality sigma=", extremality,
              ", N(0," + str(extremality) + "^2)")
        print('beta entropy is:', entropy, ' Max entropy is:', max_entropy, '\n')
        y_train = generate_sample(beta, train_size)
        for alg in algs:
            beta_zero = np.zeros(n_classes)
            y_train_lr = np.copy(y_train[:len(y_train)])
            fn = alg.name + str(n_classes) + '.ext.' + str(extremality) + '.ins.' + str(i)
            if save:
                file = path + fn
            else:
                file = None
            try:
                v = np.load(path + fn + 'kl' + '.npy')
            except:
                estimations = run_algorithm(alg, beta_zero.copy(), y_train_lr, beta, file=file)
                v = m.kl_divergence(beta, estimations)
                try:
                    if save:
                        np.save(path + fn + 'kl', v)
                except:
                    pass
            m_vals[alg.name].append(v)
    return m_vals


def three_entropy_graphs(algs, sample_length=5000, dimension=199, n_repeats=100):
    n_classes = dimension + 1
    np.random.seed(1)

    x_labels = ['High entropy', 'Medium entropy', 'Low entropy']
    y_labels = ['k=199              ']

    e1 = experiment(n_classes, sample_length, algs, n_repeats, extremality=1)
    e2 = experiment(n_classes, sample_length, algs, n_repeats, extremality=2)
    e3 = experiment(n_classes, sample_length, algs, n_repeats, extremality=3)
    labels = []
    lines1, lines2, lines3 = [], [], []
    for key in e1:
        labels.append(key)
        lines1.append(e1[key])
        lines2.append(e2[key])
        lines3.append(e3[key])

    medians1 = np.median(lines1, axis=1)
    medians2 = np.median(lines2, axis=1)
    medians3 = np.median(lines3, axis=1)
    medians = np.array([[medians1, medians2, medians3]])
    n_dots = len(medians1[0])

    low_q1 = np.percentile(lines1, 75, axis=1)
    low_q2 = np.percentile(lines2, 75, axis=1)
    low_q3 = np.percentile(lines3, 75, axis=1)
    low_qs = np.array([[low_q1, low_q2, low_q3]])

    high_q1 = np.percentile(lines1, 25, axis=1)
    high_q2 = np.percentile(lines2, 25, axis=1)
    high_q3 = np.percentile(lines3, 25, axis=1)
    high_qs = np.array([[high_q1, high_q2, high_q3]])

    x = np.arange(n_dots) * sample_length // (n_dots)
    g.plot_lines(x, medians, labels, x_labels=x_labels, y_labels=y_labels, low_lines=low_qs, high_lines=high_qs)


def mod_surfaces():
    train_size = 5000
    n_dots = 100
    n_classes = 200
    np.random.seed(1)
    n_repeats = 100
    mod_eras = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    algs = [algorithms.SNGD()]
    for eras in mod_eras:
        algs.append(algorithms.MOD(eras))

    e1 = experiment(n_classes, train_size, algs, n_repeats, extremality=1)
    e2 = experiment(n_classes, train_size, algs, n_repeats, extremality=2)
    e3 = experiment(n_classes, train_size, algs, n_repeats, extremality=3)

    x = np.arange(n_dots+1) * train_size // (n_dots+1)
    labels = []
    lines1, lines2, lines3 = [], [], []
    for key in e1:
        labels.append(key)
        lines1.append(e1[key])
        lines2.append(e2[key])
        lines3.append(e3[key])

    z1 = np.median(lines1, axis=1)
    z2 = np.median(lines2, axis=1)
    z3 = np.median(lines3, axis=1)
    z1 = np.log(z1)
    z2 = np.log(z2)
    z3 = np.log(z3)
    # N = len(algs)
    # y = np.linspace(0, 5000, N)
    mod_eras.insert(0, 1)
    y = mod_eras
    x, y = np.meshgrid(x, y)
    x_label = 'iterations'
    y_label = 'eras'
    z_label = 'log(KL)'
    title = ''
    g.plot_surface_3D(x, y, z1, x_label=x_label, y_label=y_label, z_label=z_label, title=title)
    g.plot_surface_3D(x, y, z2, x_label=x_label, y_label=y_label, z_label=z_label, title=title)
    g.plot_surface_3D(x, y, z3, x_label=x_label, y_label=y_label, z_label=z_label, title=title)


def three_entropy_three_dimensions_graph(algs, sample_length=5000, dimensions=(50, 200, 500), n_repeats=100):
    n_dots = 100
    np.random.seed(1)

    extremalities = [1, 2, 3]
    x_labels = ['High entropy', 'Medium entropy', 'Low entropy']
    y_labels = ['k=49              ', 'k=199              ', 'k=499              ']
    labels = []
    for alg in algs:
        labels.append(alg.name)

    medians = [[0,0,0],[0,0,0],[0,0,0]]
    first_quartile = [[0,0,0],[0,0,0],[0,0,0]]
    third_quartile = [[0,0,0],[0,0,0],[0,0,0]]

    for i in range(len(dimensions)):
        dim = dimensions[i]
        for j in range(len(extremalities)):
            extremality = extremalities[j]
            e = experiment(dim, sample_length, algs, n_repeats, extremality)
            lines = list(e.values())
            medians[i][j] = np.median(lines, axis=1)
            first_quartile[i][j] = np.percentile(lines, 75, axis=1)
            third_quartile[i][j] = np.percentile(lines, 25, axis=1)

    medians = np.array(medians)
    first_quartile = np.array(first_quartile)
    third_quartile = np.array(third_quartile)
    x = np.arange(n_dots+1) * sample_length // (n_dots+1)

    g.plot_lines(x, medians, labels, x_labels=x_labels, y_labels=y_labels,
                 low_lines=first_quartile, high_lines=third_quartile)



### Tests
def fast_sngd_test():
    alg1 = [algorithms.SNGD()]
    alg2 = [algorithms.SNGD2()]
    sample_length = 5000
    dimension = 199
    n_repeats = 1
    n_classes = dimension + 1
    np.random.seed(1)
    t1 = time.time()
    e1 = experiment(n_classes, sample_length, alg1, n_repeats, extremality=1, save=False)
    t2 = time.time()
    e1 = experiment(n_classes, sample_length, alg2, n_repeats, extremality=1, save=False)
    t3 = time.time()
    time1 = t2-t1
    time2 = t3-t2
    print("Algorithm ", alg1[0].name, " needed ", time1, " seconds")
    print("Algorithm ", alg2[0].name, " needed ", time2, " seconds")


def mod_is_sngd():
    n_classes = 20
    np.random.seed(1)
    algs = [algorithms.SNGD(), algorithms.MOD(1)]
    sample = np.arange(n_classes)
    beta = np.random.random(n_classes)

    hyperparams = {"a": 0.1, "b": 10000}
    betas1 = algs[0].learn(beta.copy(), hyperparams, sample)
    betas2 = algs[1].learn(beta.copy(), hyperparams, sample)
    error = np.sum(np.abs(betas1-betas2))
    print(error)


def equivalent_algs_test(alg1, alg2):
    algs = [alg1, alg2]
    name1 = alg1.name
    name2 = alg2.name
    sample_length = 500
    dimension = 99
    n_repeats = 1
    n_classes = dimension + 1
    np.random.seed(1)

    e4 = experiment(n_classes, sample_length, algs, n_repeats, extremality=4, save=False)
    E = 1
    for i in range(n_repeats):
        E = np.sum(np.abs(e4[name1][i] - e4[name2][i]))

    if E > 1e-8:
        print("Methods are not equivalent")
    else:
        print("Methods are equivalent")


# equivalent_algs_test(dsngd, csngd)
# equivalent_sngd_test()
# fast_sngd_test()
# if __name__=="__main__":
np.seterr(all='raise')


### Run the die problem to compare different algorithms

### To graph SNGD showing bad or slow convergence
three_entropy_graphs([sngd, sgd])

### To graph MEGD,SNGD, SGD
# three_entropy_graphs([megd, sngd, sgd])


### To graph MOD1000,MEGD,SNGD and SGD
# three_entropy_graphs([megd, mod1000, sngd, sgd)


### CSNGD, best MOD, SNGD and SGD
# three_entropy_graphs([csngd, sgd, mod1000, sngd])

### Run MOD with different eras and plot a surface 3D graph
mod_surfaces()

### To graph CSNGD vs SNGD
three_entropy_three_dimensions_graph([csngd, sgd, map])


