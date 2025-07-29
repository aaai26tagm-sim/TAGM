import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## Multithread
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

####################################### Global variable for recomputing true cost ####################################
# Set the following to "True" for the first run to generate the "true_cost.npy" file, 
# this is used to save the optimal cost for comparison. Set this to "False" once the cost
# is calculated, and if any problem setting changes, this needs to be set to "True" again 
# for the optimal cost to be re-calculated 

recompte_true_cost = False 

######################################################### Data set ###################################################
## Fashion-MNIST
fmnist = fetch_openml(name="Fashion-MNIST", version=1)
A_all = fmnist['data'].to_numpy().astype(np.float32) / 255.0  # (70000, 784)
y_all = fmnist['target'].astype(int).values.reshape(-1,1)

# Shuffle and train/test split
A_train, A_test, y_train, y_test = train_test_split(A_all, y_all, test_size=0.3, random_state=77)

# Standardize the data
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)

# PCA (optional)
# pca = PCA(n_components=700, whiten=True)
# A_train = pca.fit_transform(A_train)
# X_test = pca.transform(X_test)

# Reshape labels
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Total samples and feature dimension
num_samples, d = A_train.shape        

# Hot encoder 
onehot_encoder = OneHotEncoder(sparse=False, dtype=np.float32)

####################################################### Helper Function Block #################################################
# Define the projection argument for projected gradient descent
def projection(arr, lower_bound, upper_bound):
    return np.clip(np.asarray(arr), lower_bound, upper_bound)

def loss(X, Y, W, mu):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1)))) + mu/2 * np.linalg.norm(W, ord='fro') ** 2
    return loss

def gradient(X, Y, W, mu):
    """
    Y: onehot encoded 
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + mu * W
    return gd

def gradient_descent(X, Y, max_iter=670, eta=0.01, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []
 
    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W, mu))

    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W, loss_lst[-1]

class Multiclass:
    def fit(self, X, Y):
        self.loss_steps, self.W, self.true_cost = gradient_descent(X, Y)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)
    
#######################################################################################################################

################################################# True Cost Computation ###############################################
if recompte_true_cost == True:
    import time

    start_time = time.time()

    # Fit model
    model = Multiclass()
    model.fit(A_train, y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"Elapsed time: {minutes} minutes and {seconds:.2f} seconds")

    # Plot loss
    model.loss_plot()

    # Percentage of True values 
    accuracy = np.mean(model.predict(A_test) == y_test.ravel()) * 100

    print(f"Accuracy: {accuracy:.2f}%")

    ## Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Print Confusion Matrix
    conf_matrix =  confusion_matrix(y_test, model.predict(A_test))
    # Plot it
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(range(10)))
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Normalized Confusion Matrix")
    plt.show()

    ### Save the cost file 
    np.save("true_cost.npy", model.true_cost) # save the true cost

########################################################################################################################

################################################### Algorithm Block ####################################################
# Parameters for distributed implementation
N = 16 # agent number
comDelay = 1
upDelay = 1
lb = -5 # lower bound for projection 
ub = 5  # upper bound for projection 
mu = 1e-2 # regularization term

# Neighbor sets
neigh = np.tile(np.arange(0, N).reshape(-1, 1), (1, N))

K = 10 # number of classes

# Initial states 
x0 = np.zeros((d, K, N), dtype=np.float32) # all 0's across all agents 
y0 = np.zeros((d, K, N), dtype=np.float32) # all 0's across all agents 

## Hyperparameters
gamma = 1e-1 # equivalent to GM for beta = lambda = 0
gammaHB = gamma   
gammaNAG = gamma   
gammaGM = gamma 

# Momentum terms 
betaHB = 0.075 # equivalent to GM for lambda = 0 
lambdaNAG = 0.35 # equivalent to GM for beta = lambda 
lambdaGM = 0.05 
betaGM =  0.5

# Time vector (iterations)
dt = 0.1
tf = 1000 # tf x 10 = iterations 
tvec = np.arange(0, tf + dt, dt)

# Split the features equally across agents
features_per_agent = d // N   # preferably divided evenly 

# Transform
b_train = onehot_encoder.fit_transform(y_train.reshape(-1,1))

# Initialize A train list for each agent's blocks 
A_train_list = []
start_idx = 0
for i in range(N):
    end_idx = start_idx + features_per_agent
    A_i = A_train[:, start_idx:end_idx]  # shape: (n, num_features)
    A_train_list.append(A_i)
    start_idx = end_idx

# Thread-safe locks for each agent
agent_locks = [threading.Lock() for _ in range(N)]

# Initializations
x_GD = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
x_HB = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
y_HB = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
x_NAG = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
y_NAG = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
x_GM = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
y_GM = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)

true_state_GD = np.ones((len(tvec), d, K), dtype=np.float32)
true_state_HB = np.ones((len(tvec), d, K), dtype=np.float32)
true_state_NAG = np.ones((len(tvec), d, K), dtype=np.float32)
true_state_GM = np.ones((len(tvec), d, K), dtype=np.float32)

# Cost lists
cost_diffGD_lst = []
cost_diffHB_lst = []
cost_diffNAG_lst = []
cost_diffGM_lst = []

# Load true cost
true_cost = np.load("true_cost.npy").astype(np.float32)

# Probability list (1 elemnet for now, can use multiple)
probs = [0.1]

## Algorithms (GD, HB, NAG, GM) 
def update_agent_GD(cur_agent, k, x_GD, A_train, A_train_list, b_train, num_samples, mu, gamma, features_per_agent, lb, ub, projection, softmax):
    """Update single agent for Gradient Descent"""
    with agent_locks[cur_agent]:
        x_cur = x_GD[k, :, :, cur_agent].copy()
        x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]  
        A_i = A_train_list[cur_agent]
        
        Z = -A_train @ x_cur
        P = softmax(Z, axis=1)
        gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * x_block
        
        xnew_block = projection(x_block - gamma * gradif, lb, ub)
        x_GD[k+1, :, :, cur_agent] = x_cur
        x_GD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent] = xnew_block

def update_agent_HB(cur_agent, k, x_HB, y_HB, A_train, A_train_list, b_train, num_samples, mu, gammaHB, betaHB, features_per_agent, lb, ub, projection, softmax):
    """Update single agent for Heavy Ball"""
    with agent_locks[cur_agent]:
        x_cur = x_HB[k, :, :, cur_agent].copy()
        y_cur = y_HB[k, :, :, cur_agent].copy()
        
        # Descent on y first 
        x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        A_i = A_train_list[cur_agent]

        Z = -A_train @ x_cur
        P = softmax(Z, axis=1)
        gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * x_block
        
        ynew_block = projection(x_block-gammaHB*gradif+betaHB*(x_block-y_block), lb, ub)
        y_HB[k+1, :, :, cur_agent] = y_cur
        y_HB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent] = ynew_block

        # Descent on x
        y_cur_updated = y_HB[k+1, :, :, cur_agent].copy()
        y_block = y_cur_updated[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]

        Z = -A_train @ y_cur_updated
        P = softmax(Z, axis=1)
        gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * y_block
        
        xnew_block = projection(y_block-gammaHB*gradif+betaHB*(y_block-x_block), lb, ub)
        x_HB[k+1, :, :, cur_agent] = x_cur
        x_HB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent] = xnew_block

def update_agent_NAG(cur_agent, k, x_NAG, y_NAG, A_train, A_train_list, b_train, num_samples, mu, gammaNAG, lambdaNAG, features_per_agent, lb, ub, projection, softmax):
    """Update single agent for Nesterov Accelerated Gradient"""
    with agent_locks[cur_agent]:
        x_cur = x_NAG[k, :, :, cur_agent].copy()
        y_cur = y_NAG[k, :, :, cur_agent].copy()
        
        # Descent on y first 
        x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        A_i = A_train_list[cur_agent]
        
        # Momentum on x
        x_momentum = x_cur + lambdaNAG*(x_cur-y_cur)
        
        Z = -A_train @ x_momentum
        P = softmax(Z, axis=1)
        gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * x_momentum[cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :]

        ynew_block = projection(x_block-gammaNAG*gradif+lambdaNAG*(x_block-y_block), lb, ub)
        y_NAG[k+1, :, :, cur_agent] = y_cur
        y_NAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent] = ynew_block

        # Descent on x
        y_cur_updated = y_NAG[k+1, :, :, cur_agent].copy()
        y_block = y_cur_updated[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]

        # Momentum on y
        y_momentum = y_cur_updated + lambdaNAG*(y_cur_updated-x_cur)

        Z = -A_train @ y_momentum
        P = softmax(Z, axis=1)
        gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * y_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        
        xnew_block = projection(y_block-gammaNAG*gradif+lambdaNAG*(y_block-x_block), lb, ub)
        x_NAG[k+1, :, :, cur_agent] = x_cur
        x_NAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent] = xnew_block

def update_agent_GM(cur_agent, k, x_GM, y_GM, A_train, A_train_list, b_train, num_samples, mu, gammaGM, lambdaGM, betaGM, features_per_agent, lb, ub, projection, softmax):
    """Update single agent for Generalized Momentum Method"""
    with agent_locks[cur_agent]:
        x_cur = x_GM[k, :, :, cur_agent].copy()
        y_cur = y_GM[k, :, :, cur_agent].copy()
        
        # Descent on y first 
        x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        A_i = A_train_list[cur_agent]
        
        # Momentum on x
        x_momentum = x_cur + lambdaGM*(x_cur-y_cur)
        
        Z = -A_train @ x_momentum
        P = softmax(Z, axis=1)
        gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * x_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :] 
        
        ynew_block = projection(x_block-gammaGM*gradif+betaGM*(x_block-y_block), lb, ub)
        y_GM[k+1, :, :, cur_agent] = y_cur
        y_GM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent] = ynew_block

        # Descent on x
        y_cur_updated = y_GM[k+1, :, :, cur_agent].copy()
        y_block = y_cur_updated[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]

        # Momentum on y
        y_momentum = y_cur_updated + lambdaGM*(y_cur_updated-x_cur)

        Z = -A_train @ y_momentum
        P = softmax(Z, axis=1)
        gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * y_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent, :]
        
        xnew_block = projection(y_block-gammaGM*gradif+betaGM*(y_block-x_block), lb, ub)
        x_GM[k+1, :, :, cur_agent] = x_cur
        x_GM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent] = xnew_block

## Communication function
def communicate_agent_updates(cur_agent, k, neigh, N, features_per_agent, x_GD, x_HB, y_HB, x_NAG, y_NAG, x_GM, y_GM):
    """Handle communication for a single agent"""
    sendto = np.intersect1d(np.arange(N), neigh[:, cur_agent])
    sendto = sendto[sendto != cur_agent]
    
    for ii in sendto:
        with agent_locks[ii]:
            # Send to neighbors
            x_GD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, ii] = x_GD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent]
            x_HB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, ii] = x_HB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent]
            y_HB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, ii] = y_HB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent]
            x_NAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, ii] = x_NAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent]
            y_NAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, ii] = y_NAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent]
            x_GM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, ii] = x_GM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent]
            y_GM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, ii] = y_GM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, :, cur_agent]
## End of function block

# Number of worker threads 
MAX_WORKERS = min(N, 16)  

## Simulation with total ascynchrony
if upDelay == 1:
    for prob in probs:
        probCom = prob * np.ones(N)
        # initialize x0 and y0
        x_GD[0, :, :, :] = x0.copy() 
        x_HB[0, :, :, :] = x0.copy()
        y_HB[0, :, :, :] = y0.copy()
        x_NAG[0, :, :, :] = x0.copy()
        y_NAG[0, :, :, :] = y0.copy()
        x_GM[0, :, :, :] = x0.copy()
        y_GM[0, :, :, :] = y0.copy()

        for k in range(len(tvec)):
            # Coin flip
            # This is line 3 in Algorithm 1 implementing the delay in update with probabilty p
            upif = np.where(np.random.rand(N) < probCom)[0] if upDelay == 1 else np.arange(N)

            # Copy over states for agents that don't update
            for n in range(N):
                if n not in upif:
                    x_GD[k+1, :, :, n] = x_GD[k, :, :, n]
                    x_HB[k+1, :, :, n] = x_HB[k, :, :, n]
                    y_HB[k+1, :, :, n] = y_HB[k, :, :, n]
                    x_NAG[k+1, :, :, n] = x_NAG[k, :, :, n]
                    y_NAG[k+1, :, :, n] = y_NAG[k, :, :, n]
                    x_GM[k+1, :, :, n] = x_GM[k, :, :, n]
                    y_GM[k+1, :, :, n] = y_GM[k, :, :, n]
            # Parallel agent updates
            if len(upif) > 0:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Submit all agent update tasks
                    gd_futures = []
                    hb_futures = []
                    nag_futures = []
                    gm_futures = []
                    
                    # This is line 4 in Algorithm 1
                    for cur_agent in upif:
                        A_i = A_train_list[cur_agent]   
                        gd_futures.append(executor.submit(
                            update_agent_GD, cur_agent, k, x_GD, A_train, A_train_list, 
                            b_train, num_samples, mu, gamma, features_per_agent, lb, ub, 
                            projection, softmax
                        ))
                        
                        hb_futures.append(executor.submit(
                            update_agent_HB, cur_agent, k, x_HB, y_HB, A_train, A_train_list, 
                            b_train, num_samples, mu, gammaHB, betaHB, features_per_agent, lb, ub, 
                            projection, softmax
                        ))
                        
                        nag_futures.append(executor.submit(
                            update_agent_NAG, cur_agent, k, x_NAG, y_NAG, A_train, A_train_list, 
                            b_train, num_samples, mu, gammaNAG, lambdaNAG, features_per_agent, lb, ub, 
                            projection, softmax
                        ))
                        
                        gm_futures.append(executor.submit(
                            update_agent_GM, cur_agent, k, x_GM, y_GM, A_train, A_train_list, 
                            b_train, num_samples, mu, gammaGM, lambdaGM, betaGM, features_per_agent, lb, ub, 
                            projection, softmax
                        ))
                    # Wait for all updates to complete
                    for future in gd_futures + hb_futures + nag_futures + gm_futures:
                        future.result()  # This will raise any exceptions that occurred
            # Communication phase
            # Coin flip
            # this is line 10 in Algorithm 1 implementing the delay in communications with probabilty p
            comif = np.where(np.random.rand(N) < probCom)[0] if comDelay == 1 else np.arange(N)

            if len(comif) > 0:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    comm_futures = []
                    for cur_agent in comif:
                        # This is line 6 and 11 in Algorithm 1, they are combined here
                        comm_futures.append(executor.submit(
                            communicate_agent_updates, cur_agent, k, neigh, N, features_per_agent,
                            x_GD, x_HB, y_HB, x_NAG, y_NAG, x_GM, y_GM
                        ))
                    
                    # Wait for all communications to complete
                    for future in comm_futures:
                        future.result()
            else:
                # No communication - copy states
                # This is line 14 - 16 in Algorithm 1 
                for n in range(N):
                    x_GD[k+1, :, :, n] = x_GD[k, :, :, n]
                    x_HB[k+1, :, :, n] = x_HB[k, :, :, n]
                    y_HB[k+1, :, :, n] = y_HB[k, :, :, n]
                    x_NAG[k+1, :, :, n] = x_NAG[k, :, :, n]
                    y_NAG[k+1, :, :, n] = y_NAG[k, :, :, n]
                    x_GM[k+1, :, :, n] = x_GM[k, :, :, n]
                    y_GM[k+1, :, :, n] = y_GM[k, :, :, n]

            # Reconstruct true states
            for agent in range(N):
                idx_start = agent * features_per_agent
                idx_end = (agent + 1) * features_per_agent
                
                true_state_GD[k, idx_start:idx_end, :] = x_GD[k, idx_start:idx_end, :, agent]
                true_state_HB[k, idx_start:idx_end, :] = x_HB[k, idx_start:idx_end, :, agent]
                true_state_NAG[k, idx_start:idx_end, :] = x_NAG[k, idx_start:idx_end, :, agent]
                true_state_GM[k, idx_start:idx_end, :] = x_GM[k, idx_start:idx_end, :, agent]
                
            # Calculate costs 
            cost_diffGD_lst.append(loss(A_train, b_train, true_state_GD[k, :, :], mu) - true_cost)
            cost_diffHB_lst.append(loss(A_train, b_train, true_state_HB[k, :, :], mu) - true_cost)
            cost_diffNAG_lst.append(loss(A_train, b_train, true_state_NAG[k, :, :], mu) - true_cost)
            cost_diffGM_lst.append(loss(A_train, b_train, true_state_GM[k, :, :], mu) - true_cost)

            # print(f"Iteration: {k+1}")

        ## Double check
        # print("True state (GD):")
        # print(x_star)
        # print("True state (TAGD):")
        # print(true_state_GD[-1,:]) 
        # print(f"True cost is: {cost_diffGD_lst[-1]}")

## Save data
# can be loaded in another script for plotting
for prob in probs:
    prob_pct = int(prob * 100) 
    np.save(f"cost_diff_GD_prob={prob_pct}.npy", np.array(cost_diffGD_lst))
    np.save(f"cost_diff_HB_prob={prob_pct}.npy", np.array(cost_diffHB_lst))
    np.save(f"cost_diff_NAG_prob={prob_pct}.npy", np.array(cost_diffNAG_lst))
    np.save(f"cost_diff_GM_prob={prob_pct}.npy", np.array(cost_diffGM_lst))
