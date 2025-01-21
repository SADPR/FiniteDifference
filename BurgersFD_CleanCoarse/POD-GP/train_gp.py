import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import pickle
import GPy
from array import array
from matplotlib import cm
import scipy.sparse as sp

tot = time.time()
nc = 10
nt = 35
data = np.loadtxt('state.coords', delimiter=',')
q = np.array(data).T
q0 = q[0:nc, :]
q1 = q[nc:nt, :]

# mu = np.array(range(9))/8.0
# u = np.ones(9)
# q0 = np.vstack((np.hstack(np.array(range(501))/500.0 * u[np.newaxis,:].T),
#                np.hstack((np.ones(501)*mu[np.newaxis,:].T)))
#              )
# q1 = q[0:nt,:]
print('Reduced coordinates read time: ', time.time() - tot)

ns = q1.shape[1]
nTrain = 250
ixTrain = np.random.choice(range(0, ns, 3), size=nTrain, replace=False)
print('training set size: ', ixTrain.shape)
# ixTest = list(set(range(ns)) - set(ixTrain))
ixTest = list(set(range(ns)))
ixTestE = list(set(range(ns)) - set(ixTrain))
fi = open('ixTrain.npy', 'wb')
np.save(fi, ixTrain)
fi.close()
'''
ixTrain = np.load('ixTrain.npy')
ixTest = list(set(range(ns)))
ixTestE = list(set(range(ns)) - set(ixTrain))
'''

q1Pred = np.zeros_like(q1)

# Scaling factors
txmin = np.min(q0, axis=1)
txmax = np.max(q0, axis=1)
tymin = np.min(q1, axis=1)
tymax = np.max(q1, axis=1)
#tymin = np.min(q1[:, :])
#tymax = np.max(q1[:, :])

# Scale test data
txTest = q0[:, ixTest]
tyTest = q1[:, ixTest]
xTest = 2.0 * ((txTest.T - txmin) / (txmax - txmin)) - 1.0
yTest = 2.0 * ((tyTest.T - tymin) / (tymax - tymin)) - 1.0
# Scale training data
txTrain = q0[:, ixTrain]
tyTrain = q1[:, ixTrain]
xTrain = 2.0 * ((txTrain.T - txmin) / (txmax - txmin)) - 1.0
yTrain = 2.0 * ((tyTrain.T - tymin) / (tymax - tymin)) - 1.0

# Train
kernel0 = ConstantKernel(1.0e0, (1e-3, 1e2)) * \
       Matern(0.5 * np.ones(q0.shape[0]), (1e-2, 5.0), nu=1.5)
# RBF(0.5*np.ones(q0.shape[0]), (1e-2, 5.0))
# kernel0 = ConstantKernel(1.0e0, (1e-3, 1e2)) * \
#          RBF(0.5, (1e-2, 5.0))

gp = GaussianProcessRegressor(kernel=kernel0, alpha=1e-8, n_restarts_optimizer=1)
gp.fit(xTrain, yTrain)
print(gp.kernel_.get_params())

# Predict
yPred, yStd = gp.predict(xTest, return_std=True)
print('Total norm: ', np.linalg.norm(yTest - yPred) / np.linalg.norm(yTest))

# Plot
breakpoint()
for num_gp in range(nt-nc):
    plt.figure(figsize=(8, 5))
    plt.plot(np.array(range(xTest.shape[0])), yTest[:,num_gp], 'r-', label='Reference')
    plt.plot(np.array(range(xTest.shape[0])), yPred[:,num_gp], 'b.', label='GP fit')
    plt.legend(loc='upper left')
    plt.savefig('q_%d.png' %num_gp)
    plt.close()

# save
with open('gp.pkl','wb') as f:
    pickle.dump(gp,f)

q1Pred[:, ixTest] = (((yPred + 1.0) / 2.0) * (tymax - tymin) + tymin).T
print('Total norm rescaled: ', np.linalg.norm(q1 - q1Pred) / np.linalg.norm(q1))

for num_gp in range(nt-nc):
    print('Total norm rescaled GP no %d : ' % num_gp, np.linalg.norm(q1[num_gp,:] - q1Pred[num_gp,:]) / np.linalg.norm(q1[num_gp,:]))

precomp = np.linalg.solve(kernel0(xTrain)+ 1e-6 * np.eye(kernel0(xTrain).shape[0]), yTrain)
np.savetxt('precomputations.txt',precomp,fmt='%.7f')
np.savetxt('xTrain.txt',xTrain,fmt='%.7f')
np.savetxt('xTest.txt',xTest,fmt='%.7f')
np.savetxt('yPred.txt',yPred,fmt='%.7f')
with open("scalings.txt","w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in (txmin,txmax,tymin,tymax)))

print('Total GP training time: ', time.time() - tot)
