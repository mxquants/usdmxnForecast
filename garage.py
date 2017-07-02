

# Search best architecture 1


dataset.train[0].shape
_epochs = 2000
max_hidden = 5
max_neurons = 20
mse = {"hidden": [], "neurons": [], "mse": []}
hidden_vector = [None]
temp_hidden_vector = [None]
best_per_layer = []
for i in range(1, max_hidden+1):
    print("\n")
    for j in range(1, max_neurons+1):
        temp_hidden_vector[i-1] = j
        nparams, nelements, not_warn = numberOfWeights(dataset,
                                                       temp_hidden_vector)
        if not not_warn:
            print("Not viable anymore.")
            break
        hidden_vector[i-1] = j
        mse["hidden"].append(i)
        mse["neurons"].append(j)
        mlp = mx.neuralNets.mlpRegressor(hidden_layers=hidden_vector)
        mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
        mse["mse"].append(np.mean(mlp.test(dataset=dataset)["square_error"]))
        print("Evaluating: ({i},{j}) => {mse:0.6f}".format(i=i, j=j,
                                                           mse=mse["mse"][-1]))
    if not not_warn:
        break
    temp = pd.DataFrame(mse)
    min_mse_arg = temp.query("hidden == {}".format(i)).mse.argmin()
    temp_hidden_vector[i-1] = temp["neurons"].iloc[min_mse_arg]
    hidden_vector[i-1] = temp["neurons"].iloc[min_mse_arg]
    best_per_layer.append(temp["mse"].iloc[min_mse_arg])
    hidden_vector.append(None)
    temp_hidden_vector.append(None)

hidden_vector
best_per_layer

plt.plot(np.arange(len(best_per_layer))+1, best_per_layer)
plt.title("MSE best per n-neurons per hidden layer index")
plt.show()

mse_df = pd.DataFrame(mse)
mse_df
x = mse["hidden"]
y = mse["neurons"]
z = mse["mse"]
min_z, max_z = min(z), max(z)
z = [(i-min_z)/(max_z-min_z) for i in z]
plt.scatter(x, y, c=z, s=100)
# plt.gray()
plt.xlabel("Number of hidden layers")
plt.ylabel("Number of neurons at last hl")
plt.grid()
plt.show()

plt.plot(x, mse["mse"])
# plt.gray()
plt.xlabel("Number of hidden layers")
plt.ylabel("mse")
plt.grid()
plt.show()

plt.plot(y, mse["mse"], '.b')
# plt.gray()
plt.xlabel("Number of neurons")
plt.ylabel("mse")
plt.grid()
plt.show()







# Search best architecture 2
# Grid eval
grid = [np.arange(1, max_neurons) for i in range(1, max_hidden)]
grid = pd.DataFrame(grid).T


def getMSE(architecture, dataset):
    """Return MSE or inf."""
    nparams, nelements, not_warn = numberOfWeights(dataset, architecture)
    if not not_warn:
        return np.float("inf")
    mlp = mx.neuralNets.mlpRegressor(hidden_layers=architecture)
    mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
    return np.mean(mlp.test(dataset=dataset)["square_error"])


mse_res = {}
save_min = float("inf")
save_architecture = None
for i in grid:
    hidden = i
    mse_res[hidden] = []
    for j in grid[hidden].values:

        architecture = [j]*hidden
        temp_mse = getMSE(architecture, dataset)
        if temp_mse < save_min:
            save_min = temp_mse
            save_architecture = architecture
        mse_res[hidden].append(temp_mse)
        print("({}, {}) => {}".format(hidden, j, mse))

mse_vals = pd.DataFrame(mse_res, index=grid[hidden].values)
mse_vals


min_arg, min_mse = None, float("inf")
for col in mse_vals:
    local_min = mse_vals[col].min()
    local_argmin = mse_vals[col].argmin()
    if local_min < min_mse:
        min_mse = local_min
        min_arg = local_argmin
        best_architecture = (col, min_arg)
    mse_vals[col].plot()
    plt.title("Fix number of hidden layers: {}".format(col))
    plt.xlabel("Number of neurons (for each layer)")
    plt.show()

best_architecture
min_mse
