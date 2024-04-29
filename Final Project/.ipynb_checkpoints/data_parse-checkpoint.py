import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.io as sio
import jax.numpy as jnp
import sys

line_num = 0
x = []
y = []
u = []
v = []
theta = []
p = []
T = []

X_min = -4.0
X_max = 12.0
Y_min = -4.0
Y_max = 4.0

N_file = 100
DATA_PATH = "data/"

xyz_file = DATA_PATH + "uvtp_data_" + str(0) + ".csv"

with open(xyz_file, mode='r') as file:
    csvFile = csv.reader(file)
    next(csvFile, None)
    for lines in csvFile:
        x.append(float(lines[3]))
        y.append(float(lines[4]))

for i in range(0, N_file):
    file_name = DATA_PATH + "/uvtp_data_" + str(i) + ".csv"
    time = 0.2 * i
    T.append(time)
    with open(file_name, mode='r') as file:
        csvFile = csv.reader(file)
        next(csvFile, None)
        for lines in csvFile:
            u.append(float(lines[8]))
            v.append(float(lines[9]))
            theta.append(float(lines[7]))
            p.append(float(lines[2]))
    print(f"File: {i} is Done.")

x = np.array(x)
y = np.array(y)
u = np.array(u)
u = u.reshape(N_file, x.shape[0])
v = np.array(v)
v = v.reshape(N_file, x.shape[0])
theta = np.array(theta)
theta = theta.reshape(N_file, x.shape[0])
p = np.array(p)
p = p.reshape(N_file, x.shape[0])
T = np.array(T)
id_mask = []
id_bc = []
for i in range(np.shape(x)[0]):
    if ((x[i] >= X_min) and (x[i] <= X_max)) and ((y[i] >= Y_min) and (y[i] <= Y_max)):
        id_mask.append(i)

x_sub = x[id_mask]
y_sub = y[id_mask]
print(x_sub.shape)

u_sub = u[:, id_mask]
v_sub = v[:, id_mask]
p_sub = p[:, id_mask]
theta_sub = theta[:, id_mask]

# If temperature has to be reported Positive
# id_neg = np.where(theta_sub < 0)
# print("Prior", id_neg)
# theta_sub[id_neg] = 0.0
# id_neg_test = np.where(theta_sub < 0)
# print("Post", theta_sub)
theta_ic = theta_sub[0, :]

# fig = plt.figure()
# ax = fig.add_subplot()
# c = ax.scatter(x_bc, y_bc, c=theta_bc[50, :], cmap='hsv')
# plt.show()

x_min = np.min(x_sub)
x_max = np.max(x_sub)
y_min = np.min(y_sub)
y_max = np.max(y_sub)

for i in range(np.shape(x)[0]):
    if (x[i] == x_min) and ((y[i] >= y_min) and (y[i] <= y_max)):
        id_bc.append(i)
    if (x[i] == x_max) and ((y[i] >= y_min) and (y[i] <= y_max)):
        id_bc.append(i)
    if ((x[i] >= x_min) and (x[i] <= x_max)) and (y[i] == y_min):
        id_bc.append(i)
    if ((x[i] >= x_min) and (x[i] <= x_max)) and (y[i] == y_max):
        id_bc.append(i)

x_bc = x[id_bc]
y_bc = y[id_bc]

u_bc = u[:, id_bc]
v_bc = v[:, id_bc]
p_bc = p[:, id_bc]

theta_bc = theta[:, id_bc]

## Cyl Boundary
x_cyl = np.array([0.5 * np.cos(angle) for angle in range(0, 360)])
y_cyl = np.array([0.5 * np.sin(angle) for angle in range(0, 360)])
x_cyl = x_cyl.flatten()[:, None]
y_cyl = y_cyl.flatten()[:, None]
xy_cyl = np.hstack((x_cyl, y_cyl))
N_cyl = xy_cyl.shape[0]
t_cyl = np.array(T).flatten()[:, None]

xy_cyl_tile = jnp.tile(xy_cyl, (N_file, 1))
t_cyl_tile = jnp.tile(t_cyl.T, (N_cyl, 1))
t_cyl_tile = t_cyl_tile.swapaxes(0, 1)
t_cyl_tile = t_cyl_tile.reshape(N_cyl * N_file, 1)

xyt_cyl = jnp.hstack((xy_cyl_tile, t_cyl_tile))
sh = xyt_cyl.shape
theta_cyl = 1.0 * jnp.ones(shape=(sh[0], 1))
print(theta_cyl)

data = {"x": x_sub, "y": y_sub, "Time": T,
        "u": u_sub, "v": v_sub, "theta": theta_sub,
        "p": p_sub, "x_ic": x_sub,
        "y_ic": y_sub, "theta_ic": theta_ic,
        "x_bc": x_bc, "y_bc": y_bc,
        "theta_bc": theta_bc, "p_bc": p_bc, "xyt_cyl": xyt_cyl,
        "theta_cyl": theta_cyl}

sio.savemat("data.mat", data)
print(data.keys())

fig = plt.figure()
ax = fig.add_subplot(121)
c = ax.scatter(data["x"], data["y"], c=data["theta"][1, :], cmap='hsv')
plt.colorbar(c)
# c = ax.scatter(x_cyl, y_cyl)
plt.legend()

plt.show()
