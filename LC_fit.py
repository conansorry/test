# from lmfit import model
from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error
import math

def impd( x, l1, c1 ):
    z1 = x * l1 - 1.0 / (x * c1)
    return  z1


def impdy( x, l1, c1 ):
    z1 = x * l1 - 1.0 / (x * c1)

    return -1.0/z1



def impd2( x, l1, l2, c1, c2 ):
    z1 = x * l1 - 1.0 / (x * c1)
    z2 = x * l2 - 1.0 / ( x * c2) + 1.0e-2
    return  z1 * z2 / (z1 + z2)

def impd2y( x, l1, l2, c1, c2 ):
    z1 = x * l1 - 1.0 / (x * c1)
    z2 = x * l2 - 1.0 / ( x * c2) + 1.0e-2
    return  -(1.0/z1 + 1.0/z2)

def impd_cp( x, l1, l2, l12, p1, p2, p12 ):
    ll12 = math.sqrt(l1 * l2)*l12
    # print(p1 , p2 , p12)
    pp12 = math.sqrt(p1 * p2) * p12
    z1 = x * l1 + x * ll12 - p1 / (x) - pp12 / (x )
    z2 = x * l2 + x * ll12 - p2 / (x) -  pp12 / (x )+ 1.0e-2
    return  z1 * z2 / (z1 + z2)


def residual1( pars, x, data):
    l1 = pars['L1']
    c1 = pars['C1']

    return impd( x, l1, c1 ) - data


def residual_y1( pars, x, data):
    l1 = pars['L1']
    c1 = pars['C1']

    return impdy( x, l1, c1) - data


def residual( pars, x, data):
    l1 = pars['L1']
    l2 = pars['L2']

    c1 = pars['C1']
    c2 = pars['C2']

    return impd2( x, l1, l2, c1, c2 ) - data


def residual_y( pars, x, data):
    l1 = pars['L1']
    l2 = pars['L2']

    c1 = pars['C1']
    c2 = pars['C2']

    return impd2y( x, l1, l2, c1, c2 ) - data

def residual_s( pars, x, data):

    l1 = pars['L1']
    l2 = pars['L2']
    l12 = pars['L12']

    p1 = pars['P1']
    p2 = pars['P2']
    p12 = pars['P12']
    return impd_cp( x, l1, l2, l12, p1,p2, p12) - data


def fit_1st(i):
    pfit = Parameters()
    pfit.add(name='L1', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)

    pfit.add(name='C1', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)

    data = np.loadtxt(res_p + name)
    Z_data = data[:, 4]

    x = 2.0 * 3.14159 * data[:, 0] / W_nor

    mini = Minimizer(residual1, pfit, fcn_args=(x, Z_data))
    out = mini.leastsq()

    best_fit = Z_data + out.residual
    report_fit(out.params)


    pfit2 = Parameters()
    pfit2.add(name='L1', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)

    pfit2.add(name='C1', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)

    Y_data = -1.0 / Z_data
    mini2 = Minimizer(residual_y1, pfit2, fcn_args=(x, Y_data))
    out2 = mini2.leastsq()

    best_fit2 = Y_data + out2.residual

    report_fit(out2.params)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    e1 = mean_absolute_error(best_fit, Z_data)
    ax[0].plot(data[:, 0], Z_data)
    ax[0].plot(data[:, 0], best_fit, '--', label='best fit')
    ax[0].set_title(str(e1))

    e2 = mean_absolute_error(1.0 / best_fit2, 1.0 / Y_data)
    ax[1].plot(data[:, 0], Y_data)
    ax[1].plot(data[:, 0], best_fit2, '--', label='best lc')
    ax[1].set_title(str(e2))

    fig.suptitle(str(i))

    return [i, out.params['L1'], out.params['C1'],
                out2.params['L1'], out2.params['C1'],
                e1, e2]



def fit_2nd(i):
    pfit = Parameters()
    pfit.add(name='L1', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)
    pfit.add(name='L2', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)

    pfit.add(name='C1', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)
    pfit.add(name='C2', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)

    data = np.loadtxt(res_p + name)
    Z_data = data[:, 4]

    x = 2.0 * 3.14159 * data[:, 0] / W_nor

    mini = Minimizer(residual, pfit, fcn_args=(x, Z_data))
    out = mini.leastsq()

    best_fit = Z_data + out.residual
    report_fit(out.params)


    pfit2 = Parameters()
    pfit2.add(name='L1', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)
    pfit2.add(name='L2', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)

    pfit2.add(name='C1', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)
    pfit2.add(name='C2', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)

    Y_data = -1.0 / Z_data
    mini2 = Minimizer(residual_y, pfit2, fcn_args=(x, Y_data))
    out2 = mini2.leastsq()

    best_fit2 = Y_data + out2.residual

    report_fit(out2.params)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    e1 = mean_absolute_error(best_fit, Z_data)
    ax[0].plot(data[:, 0], Z_data)
    ax[0].plot(data[:, 0], best_fit, '--', label='best fit')
    ax[0].set_title(str(e1))

    e2 = mean_absolute_error(1.0 / best_fit2, 1.0 / Y_data)
    ax[1].plot(data[:, 0], Y_data)
    ax[1].plot(data[:, 0], best_fit2, '--', label='best lc')
    ax[1].set_title(str(e2))

    fig.suptitle(str(i))

    return [i, out.params['L1'], out.params['C1'], out.params['L2'], out.params['C2'],
                out2.params['L1'], out2.params['C1'], out2.params['L2'], out2.params['C2'],
                e1, e2]
# gmodel = Model()
W_nor = 1.0e9
path = "D:\Dropbox\Python Program\Forward Interpretor\\test\\"

res_p = path + "S2YZ\\"
sv_p = path + "LC_fit\\1st_order\\"

if not os.path.exists(sv_p): os.mkdir(sv_p)
res = []
for i in range(200):
    print(i)
    name = "YZ_"+str(i) + ".txt"

    # res.append(fit_2nd(i))
    res.append(fit_1st(i))
    plt.savefig( sv_p + "Fit_" + str(i) + ".png")

np.savetxt(sv_p + "LC_Summery.csv", res, fmt='%.5e', delimiter=', ' )

    # plt.legend()
    # plt.show()
