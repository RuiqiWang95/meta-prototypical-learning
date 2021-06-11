import os
import shutil
import time
import pprint
import numpy as np
import scipy.stats


_utils_pp = pprint.PrettyPrinter()


def mean_confidence_interval(accs, confidence=0.95):
    accs = np.array(accs)
    # m = np.mean(accs)
    # print(type(accs))
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)
    print('\n')


def pprint(x):
    _utils_pp.pprint(x)

