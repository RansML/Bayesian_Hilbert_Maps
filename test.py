import numpy as np
import pandas as pd
import sbhm
import matplotlib.pyplot as pl


def load_parameters(case):
    parameters = \
        {'gilad2': \
             ('Dataset/gilad2.csv',
              (50, 50),
              (-300, 300, 0, 300),
              1,
              0.3,
              0.075
              ),

         'intel': \
             ('Dataset/intel.csv',
              (5, 5),
              (-20, 20, -25, 10),
              1,
              0.01,
              6.71
              ),

         }

    return parameters[case]


fn_train, cell_resolution, cell_max_min, skip, thresh, gamma = load_parameters('intel')

#read data
g = pd.read_csv(fn_train, delimiter=',').values
print('shapes:', g.shape)
X_train = np.float_(g[:, 0:3])
Y_train = np.float_(g[:, 3][:, np.newaxis]).ravel() #* 2 - 1

max_t = len(np.unique(X_train[:, 0]))
for ith_scan in range(0, 1):  # max_t, skip):

    # extract data points of the ith scan
    ith_scan_indx = X_train[:, 0] == ith_scan
    print('{}th scan:\n  N={}'.format(ith_scan, np.sum(ith_scan_indx)))
    X_new = X_train[ith_scan_indx, 1:]
    y_new = Y_train[ith_scan_indx]

    if ith_scan == 0:
        # get all data for the first scan and initialize the model
        X, y = X_new, y_new
        bhm_mdl = sbhm.SBHM(gamma=gamma, grid=None, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=X,
                            calc_loss=False)
    else:
        # information filtering
        q_new = bhm_mdl.predict_proba(X_new)[:, 1]
        info_val_indx = np.absolute(q_new - y_new) > thresh
        X, y = X_new[info_val_indx, :], y_new[info_val_indx]
        print('  {:.2f}% points were used.'.format(X.shape[0] / X_new.shape[0] * 100))

    # training
    bhm_mdl.fit(X, y)

    # query the model
    q_resolution = 0.5
    xx, yy = np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                         np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
    q_x = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
    q_mn = bhm_mdl.predict_proba(q_x)[:, 1]

    # plot
    pl.figure(figsize=(13, 5))
    pl.subplot(121)
    ones_ = np.where(y == 1)
    pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5, edgecolors='')
    pl.title('Laser hit points at t={}'.format(ith_scan))
    pl.xlim([cell_max_min[0], cell_max_min[1]]);
    pl.ylim([cell_max_min[2], cell_max_min[3]])
    pl.subplot(122)
    pl.title('SBHM at t={}'.format(ith_scan))
    pl.scatter(q_x[:, 0], q_x[:, 1], c=q_mn * 2 - 1, cmap='jet', s=10, marker='8', edgecolors='')
    # pl.imshow(q_mn.reshape(xx.shape)*2-1)
    pl.colorbar()
    pl.xlim([cell_max_min[0], cell_max_min[1]]);
    pl.ylim([cell_max_min[2], cell_max_min[3]])
    #pl.savefig('Output/step' + str(ith_scan) + '.png', bbox_inches='tight')
    pl.show()
    # pl.close("all")