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


def residual1( pars, x, data):
    l1 = pars['L1']
    c1 = pars['C1']

    return impd( x, l1, c1 ) - data


def residual_y1( pars, x, data):
    l1 = pars['L1']
    c1 = pars['C1']

    return impdy( x, l1, c1) - data





def fit_1st(i, sv_p, W_nor):
    load_name = "YZ_" + str(i) + ".txt"

    pfit = Parameters()
    pfit.add(name='L1', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)

    pfit.add(name='C1', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)

    data = np.loadtxt(res_p + load_name)
    Z_data = data[:, 4]

    x = 2.0 * 3.14159 * data[:, 0] / W_nor

    mini = Minimizer(residual1, pfit, fcn_args=(x, Z_data))
    out = mini.leastsq()

    best_fit = Z_data + out.residual
    # report_fit(out.params)


    pfit2 = Parameters()
    pfit2.add(name='L1', value=1.0e-9 * W_nor, min=0, max=10.0e-2 * W_nor, vary=True)

    pfit2.add(name='C1', value=1.0e-15 * W_nor, min=0, max=1.0e-9 * W_nor, vary=True)

    Y_data = -1.0 / Z_data
    mini2 = Minimizer(residual_y1, pfit2, fcn_args=(x, Y_data))
    out2 = mini2.leastsq()

    best_fit2 = Y_data + out2.residual

    # report_fit(out2.params)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    e1 = mean_absolute_error(best_fit, Z_data)
    e2 = mean_absolute_error(1.0 / best_fit2, 1.0 / Y_data)

    ax[0].plot(data[:, 0], Z_data)
    ax[0].plot(data[:, 0], best_fit, '--', label='best fit')
    ax[0].set_title(str(e1))


    ax[1].plot(data[:, 0], Y_data)
    ax[1].plot(data[:, 0], best_fit2, '--', label='best lc')
    ax[1].set_title(str(e2))

    fig.suptitle(str(i))
    plt.savefig(sv_p + "Fit_" + str(i) + ".png")
    plt.close(fig)

    if( e1 < e2 ):
        return [out.params['L1'].value, out.params['C1'].value,e1, out.params['L1'].stderr, out.params['C1'].stderr]
    else:
        return [out2.params['L1'].value, out2.params['C1'].value, e2, out2.params['L1'].stderr, out2.params['C1'].stderr]

# gmodel = Model()
W_nor = 1.0e9
path = "D:\Dropbox\Python Program\Forward Interpretor\\test\\"

res_p = path + "S2YZ\\"
sv_p = path + "LC_fit\\1st_order\\"
pp_p = path + "LC_fit\\"

if not os.path.exists(sv_p): os.mkdir(sv_p)
if not os.path.exists(pp_p): os.mkdir(pp_p)
print()
res = np.zeros((100,6))
index = []
ct = 0
for i in range(200):
    print(i)


    test = fit_1st(i, sv_p, W_nor)

    res[i%100, 0]= i
    res[i % 100, 1:6] = test[:]
    if( test[2] < 10 ):
        ct = ct + 1
        index.append(i)

    if( (i+1)%100 == 0 ):
        pth = pp_p + "LC_P"+str(int(i/100))+".txt"
        np.savetxt(pth, res, fmt='%.8e')
        res = np.zeros((100, 6))
        print("total good case:", ct)

np.savetxt(pp_p + "Good_Index.txt", index, fmt='%8i')



    # plt.legend()
    # plt.show()
