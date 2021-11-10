import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smoother( res_in, cut_f = 0 ):

    res_out = np.zeros(np.shape(res_in))
    for a in range(len(res_in[-1,:])):
        x = res_in[:,a]
        x2 = savgol_filter(x, 51, 3)
        # if( a > 0):
            # plt.plot(x)
            # plt.plot(x2)
        res_out[:, a] = x2.real
    # plt.show()

    return res_out

for i in range(200):
    data_p = "D:\Dropbox\Python Program\Forward Interpretor\\test\Res\Res_"+str(i)+".txt"
    data_p_sv = "D:\Dropbox\Python Program\Forward Interpretor\\test\Smooth\Res_Smooth_"+str(i)+".txt"
    res_in = np.loadtxt(data_p)
    res_o = smoother(res_in)
    np.save(data_p_sv, res_o)

