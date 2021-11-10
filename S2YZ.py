import numpy as np
import matplotlib.pyplot as plt
import os

def s2yz( data, sv_p, plt_p, res_name):
    # find the resonance from Re and Im of S parameter file. Return

    yz = np.zeros([len(data[:,0]), 5])
    yz[:,0] = data[:,0]
    for i in range(len(data[:,0])):
        impd_y = -2.0*complex(data[i,1], data[i,2])/complex(data[i,5], data[i,6])/377.0
        impd_z = 1.0/impd_y

        yz[i, 1] = impd_y.real
        yz[i, 2] = impd_y.imag

        yz[i,3] = impd_z.real
        yz[i,4] = impd_z.imag

    np.savetxt(sv_p + res_name + ".txt", yz)

    fig, axi = plt.subplots(nrows=1, ncols=2)
    axi[0].plot(yz[:, 0], yz[:, 1])
    axi[0].plot(yz[:, 0], yz[:, 2])

    axi[1].plot(yz[:, 0], yz[:, 3])
    axi[1].plot(yz[:, 0], yz[:, 4])
    axi[0].set_title(res_name)
    fig.savefig(plt_p + res_name + ".png")
    plt.close(fig)

    return max(data[:,3]*data[:,3] + data[:,4] * data[:,4])

path = "D:\Dropbox\Python Program\Forward Interpretor\\test\\"
res_p = path+ "Smooth\\"
sv_p = path+ "S2YZ\\"
plt_p = path+"S2YZ_plt\\"

if not os.path.exists(sv_p): os.mkdir(sv_p)
if not os.path.exists(plt_p): os.mkdir(plt_p)
n_case = 200
properties = np.zeros((n_case,2))


for i in range(200):
    print(i)

    properties[i,0] = i
    res_in = np.loadtxt(res_p + "Res_Smooth_" + str(i) + ".txt")

    res_name = "YZ_" + str(i)
    properties[i,1] = s2yz(res_in, sv_p, plt_p, res_name)

np.savetxt(path + "properties.txt",  properties, fmt='%.5e')
