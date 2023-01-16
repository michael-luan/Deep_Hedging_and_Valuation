# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:51:04 2022

@author: micha
"""

import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from GOU_Simulator import Simulator

from scipy.stats import norm

# option contract parameters
T = 1/4
K1 = 9.5
K2 = 10.5  

# market environment parameters
sigma = 0.4
r = 0.02
mu = 0.05
kappa = 2
theta = np.log(10)
S0 = 10
trans_fee = 0.005

# trading parameters
NdT = 90
t = np.linspace(0,T,NdT)
dt = t[1]-t[0]     


#%% bullspread black scholes valuation and delta

def CallPrice(S,K,tau,sigma,r):
    dp = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    dm = (np.log(S/K)+(r-0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    
    return S*norm.cdf(dp) - K*np.exp(-r*tau)*norm.cdf(dm)

def bullspreadPrice(S, K1, K2, tau, sigma, r):
    return CallPrice(S, K1, tau, sigma, r) - CallPrice(S, K2, tau, sigma, r)

###############################################################################

def CallDelta(S,K,tau,sigma,r):
    tau+=1e-10

    dp = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    
    return norm.cdf(dp)

def bullspreadDelta(S, K1, K2, tau, sigma, r):
    return CallDelta(S, K1, tau, sigma, r) - CallDelta(S, K2, tau, sigma, r)

###############################################################################

def Hedge(S, alpha):
    
    C0 = bullspreadPrice(S0, K1, K2, T, sigma, r)
    transaction = abs(alpha[:, 0 , 0]) * trans_fee
    # start the bank account with value of contract and purchasing initial shares
    
    
    bank = C0 - alpha[:,0,0]*(S[:,0]) - transaction
    
    for i in range(NdT-1):
        
        # accumulate bank account to next time step
        bank *= np.exp(r*dt)
        
        # rebalance the position
        transaction = abs(alpha[:, i + 1, 0] - alpha[:, i, 0]) * trans_fee
        bank -= (alpha[:,i+1,0]-alpha[:,i,0]) * S[:,i+1] + transaction
        
    # liquidate terminal assets, and pay what you owe from the contract
    # here, we short the call at K1 and we long the call at K2
    bank += alpha[:,-1,0]*S[:,-1] - ((S[:,-1]-K1)*(S[:,-1]>K1) - (S[:,-1]-K2)*(S[:,-1]>K2))- 0.5
    
    return bank


###############################################################################

def sim_valuation(nsims = 10_000):
    
    model = Simulator()
    S = model.Sim(10_000)
    S = torch.tensor(S)
    # print("T", (T - np.matlib.repmat(t, nsims, 1)).shape)
    alpha_BS = torch.unsqueeze(
        torch.tensor(bullspreadDelta(S.detach().numpy(), K1, K2, T - np.matlib.repmat(t, nsims, 1), sigma, r)), dim=2)
    bank_BS = Hedge(S, alpha_BS)
    return bank_BS



PnL_BS = sim_valuation()

plt.hist(PnL_BS.detach().numpy(),bins = 50,  alpha=0.6,color="skyblue", ec="blue")
plt.xlabel('P&L',fontsize=16)
plt.ylabel('Freq.',fontsize=16)
plt.title("Black-Scholes Hedging Strategy P&L", fontsize = 16)




#%% ANN setup

class HedgeNet(nn.Module):
    
    def __init__(self, nNodes, nLayers ):
        super(HedgeNet, self).__init__()
        
        # single hidden layer
        self.prop_in_to_h = nn.Linear(2, nNodes)
        
        self.prop_h_to_h = []
        for i in range(nLayers-1):
            self.prop_h_to_h.append(nn.Linear(nNodes, nNodes))
            
        self.prop_h_to_out = nn.Linear(nNodes, 1)

    def forward(self, x):
        
        # input into  hidden layer
        h = torch.sigmoid(self.prop_in_to_h(x))
        
        for prop in self.prop_h_to_h:
            h = torch.relu(prop(h))
        
        # hidden layer to output layer - no activation
        y = self.prop_h_to_out(h)
        
        return y
    
    def parameters(self):
        
        params = list(self.prop_in_to_h.parameters())
        for prop in self.prop_h_to_h:
            params += list(prop.parameters())
            
        params += list(self.prop_h_to_out.parameters())
        
        return params

def RunHedge(S, alpha):
    

    
    # start the bank account with value of contract and purchasing initial shares
    bank = - alpha[:,0,0]*S[:,0] - abs(alpha[:,0,0] * trans_fee)
    
    for i in range(NdT-1):
        
        # accumulate bank account to next time step
        bank *= np.exp(r*dt)
        
        # rebalance the position
        bank -= (alpha[:,i+1,0]-alpha[:,i,0]) * S[:,i+1] + abs(alpha[:,i+1,0]-alpha[:,i,0])*trans_fee
        
    # liquidate terminal assets, and pay what you owe from the contract
    bank += alpha[:,-1,0]*S[:,-1] - (S[:,-1]-K1)*(S[:,-1]>K1) + (S[:,-1]-K2)*(S[:,-1]>K2)
    
    
    return bank




def Sim(net, nsims = 10_000):

    # simulate the asset price
    model = Simulator()
    S = model.Sim(nsims)
    S = torch.tensor(S)
    
    # combine the features into a tensor of dimension nsims x ndt x 2 
    x = torch.zeros((nsims, NdT,2))
    
    ###### feature 1: time
    x[:,:,0] = torch.tensor(2*t/T-1).float().repeat(nsims,1)
    
    ###### feature 2: asset price
    
    # print(x[:,:,1].shape)
    # print((2*S/S0-1).shape)
    
    x[:,:,1] = 2*S/S0-1
    
    
    # push the x values through the ANN -- the last dimension is treated as the features
    alpha = net(x)
    bank = RunHedge(S, alpha)
    
    # run the Black-Scholes hedge for comparison
    
    return bank

#%% plotting PnL
    
def Plot_Strategy(net, epoch = "0"):
    
    t = [0,0.5*T,0.9*T]
    S = np.linspace(0.75*K2,1.25*K2,51)
    for i, time in enumerate(t):
        
        plt.subplot(1,3,i+1)
        plt.title('t=' + str(time),fontsize=16)
        
        plt.plot(S, bullspreadDelta(S, K1, K2, T-time, sigma, r), color = "skyblue")
        
        x = torch.zeros((len(S),2))
        x[:,0] = 2*time/T-1
        x[:,1] = torch.tensor(2*S/S0-1)
        
        alpha = net(x)
        
        plt.plot(S, alpha.detach().numpy(), color = "orange")
        
        if i == 1:
            plt.xlabel('S', fontsize=16)
        if i == 0:
            plt.ylabel(r'$\Delta(S)$', fontsize=16)
        if i == 2:
            plt.legend(["Black-Scholes", "Network Estimate"])
        
        plt.ylim(0,1)
        
    plt.suptitle(r"$\Delta(S)$ of BS and ANN Strategies, Epoch " + epoch, fontsize = 16)
    plt.show()
    
    
    

def Plot_PnL(loss_hist, net, name="0"):
    plt.figure(figsize=(10,6))
    plt.suptitle("ANN Loss History And PnL Distribution of Terminal Strategy, Epoch: " + name, fontsize = 16)
    plt.subplot(1,2,1)
    plt.plot(loss_hist, color = "red")
    plt.xlabel('iteration',fontsize=16)
    plt.ylabel('loss',fontsize=16)

    plt.subplot(1,2,2)
    PnL = Sim(net, 10_000)
    PnL_BS = sim_valuation(10_000)
    plt.hist(PnL_BS.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.6,color="skyblue", ec="blue",label="Black-Scholes")
    plt.hist(PnL.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.6,color="orange",ec="black",label="Network Prediction")    
    plt.xlabel('P&L',fontsize=16)
    plt.ylabel('Count',fontsize=12)
    plt.legend()
    
    plt.tight_layout(pad=2)
    plt.show()



# no loss history so the left graph should be zero    


#%% conditional values at risk

def CVaR(sample, quantile):
    # requires sample of a dist and the percent quantile out of 1
    quantile = torch.tensor([quantile])
    percent = np.quantile(sample.detach().numpy(), quantile)
    cvar = sample[sample.detach().numpy() <= percent].mean()
    return cvar
        

#%% ANN training 3 layers of 50 nodes

def FitNet(name, net, Nepochs = 10_000):
    
    mini_batch_size = 100
    
    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    loss_hist = []
    
    Plot_PnL(loss_hist, net)
    Plot_Strategy(net)
    
    for epoch in range(Nepochs+1):  # loop over the dataset multiple times

        # grab a mini-batch from simulations
        PnL = Sim(net, mini_batch_size)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # set loss function to be conditional value at risk of 10%
        loss = -CVaR(PnL, 0.1)

        # propogate the sensitivity of the output to the model parameters 
        # backwards through the computational graph
        loss.backward()

        # update the weights and biases by taking a SGD step
        optimizer.step()

        # store running loss
        loss_hist.append(loss.item())
        
        if ((epoch) % 500 == 0 and (epoch>10) ):
            print(epoch)
            Plot_PnL(loss_hist, net, str(epoch) + " " + name)
            Plot_Strategy(net, str(epoch) + " " + name)
    
    print(epoch)
    Plot_PnL(loss_hist, net, str(Nepochs))
    Plot_Strategy(net)
    print('Finished Training')

    
    return loss_hist

#%% Changing the architecture: Switching up layers using ADAM

net = HedgeNet(50,3)
for param in net.parameters():
    print(type(param.data), param.size())

loss_hist_1L = FitNet(", 3 Layers of 50 Nodes, ADAM", net, 5000)


net1 = HedgeNet(50,1)
loss_hist = FitNet("Single Layer of 50 Nodes, ADAM", net1, 5000)


net2 = HedgeNet(50, 10)
loss_hist_10L = FitNet(", 10 Layers of 50 Nodes, ADAM", net2, 5000)



net3 = HedgeNet(5, 5)
loss_hist_simple = FitNet(", 5 Layers of 5 Nodes, ADAM", net3, 5000)

print((loss_hist_1L[-1]-0.02)*np.exp(-r*T))
print((loss_hist[-1]-0.02)*np.exp(-r*T))
print((loss_hist_10L[-1]-0.02)*np.exp(-r*T))
print((loss_hist_simple[-1]-0.02)*np.exp(-r*T))


#%% Plotting loss history


losses = [loss_hist_1L, loss_hist, loss_hist_10L, loss_hist_simple]
plt.rcParams['figure.figsize'] = [10, 8]
fig, axes = plt.subplots(4,1, sharex = True)
titles = ["3 Layers 50 Nodes", "1 Layer 50 Nodes", "10 Layers 50 Nodes", "5 Layers 5 Nodes"]
colours = ["red", "blue", "green", "purple"]
CVaRs = ["0.58", "0.74", "0.84", "0.59"]
locs = [1,4,4,1]

for idx_row, row in enumerate(axes):        
    row.set_ylim(0.5, 1)
    if idx_row == 1:
        row.set_ylabel("Loss", fontsize = 16, x = 0.3)
        
    row.plot(losses[idx_row], label = r"$CVaR_{0.1}="+  CVaRs[idx_row]+"$", color = colours[idx_row])
    
    row.legend(loc = locs[idx_row], fontsize = 12)
    
    
    row.set_title(titles[idx_row], fontsize = 15, y = 0.97)

fig.suptitle("Loss History Based on Layer and Node Architecture", fontsize = 16, y = 0.95)


#%% SGD

def FitNetSGD(name, net, Nepochs = 10_000):
    
    mini_batch_size = 100
    
    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    loss_hist = []
    
    Plot_PnL(loss_hist, net)
    Plot_Strategy(net)
    
    for epoch in range(Nepochs+1):  # loop over the dataset multiple times

        # grab a mini-batch from simulations
        PnL = Sim(net, mini_batch_size)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # set loss function to be conditional value at risk of 10%
        loss = -CVaR(PnL, 0.1)

        # propogate the sensitivity of the output to the model parameters 
        # backwards through the computational graph
        loss.backward()

        # update the weights and biases by taking a SGD step
        optimizer.step()

        # store running loss
        loss_hist.append(loss.item())
        
        if ((epoch) % 500 == 0 and (epoch>10) ):
            print(epoch)
            Plot_PnL(loss_hist, net, str(epoch) + " " + name)
            Plot_Strategy(net, str(epoch) + " " + name)
    
    print(epoch)
    Plot_PnL(loss_hist, net, str(Nepochs))
    Plot_Strategy(net)
    print('Finished Training')

    return loss_hist


netSGD = HedgeNet(50, 3)
loss_hist_sgd = FitNet(", 3 Layers of 50 Nodes, SGD", netSGD, 5000)


#%%

fig, axes = plt.subplots()
axes.plot(loss_hist_1L, color = "red", label = r"ADAM, $CVaR_{0.1} = 0.584$", alpha = 0.5)
axes.plot(loss_hist_sgd, color = "blue", label = r"SGD, $CVaR_{0.1} = 0.582$", alpha = 0.5)

axes.legend(loc = 1, fontsize = 16)
axes.set_xlim(-10, 1000)
axes.set_ylim(0.5, 1)
axes.set_ylabel(r"$\ell({\theta}) : CVaR_{0.1}$", fontsize = 16)
axes.set_xlabel("Epoch", fontsize = 16)
fig.suptitle("Loss History of ADAM and SGD Optimizer First 1000 Epochs", fontsize = 16, y = 0.95)

#%% ANN with lag 1

class HedgeNet(nn.Module):
    
    def __init__(self, ninputs, nNodes, nLayers):
        super(HedgeNet, self).__init__()
        
        # single hidden layer
        self.prop_in_to_h = nn.Linear( ninputs, nNodes)
        
        # nLayers-1 hidden layers
        self.prop_h_to_h = nn.ModuleList([nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])
            
        self.prop_h_to_out = nn.Linear(nNodes, 1)

    def forward(self, x):
        
        # input into  hidden layer
        h = torch.sigmoid(self.prop_in_to_h(x))
        
        for prop in self.prop_h_to_h:
            h = torch.relu(prop(h))
        
        # hidden layer to output layer - no activation
        y = self.prop_h_to_out(h)
        
        return y
    
    def parameters(self):
        
        params = list(self.prop_in_to_h.parameters())
        for prop in self.prop_h_to_h:
            params += list(prop.parameters())
            
        params += list(self.prop_h_to_out.parameters())
        
        return params

def SimLag1(net, nsims = 10_000):

    # simulate the asset price
    model = Simulator()
    S = model.Sim(nsims)
    S1 = np.c_[S[:,0], S[:,:-1]]
    S = torch.tensor(S)
    S1 = torch.tensor(S1)
    
    # combine the features into a tensor of dimension nsims x ndt x 3
    x = torch.zeros((nsims, NdT,3))
    
    ###### feature 1: time
    x[:,:,0] = torch.tensor(2*t/T-1).float().repeat(nsims,1)
    
    ###### feature 2: asset price
    
    # print(x[:,:,1].shape)
    # print((2*S/S0-1).shape)
    
    x[:,:,1] = torch.tensor(2*S/S0-1)
    
    ###### feature 3: asset price with lag 1
    
    x[:,:,2] = torch.tensor(2*S1/S0-1)
    
    
    # push the x values through the ANN -- the last dimension is treated as the features
    alpha = net(x)
    bank = RunHedge(S, alpha)
    
    # run the Black-Scholes hedge for comparison
    
    return bank

def FitNet1(net, Nepochs = 10_000):
    
    mini_batch_size = 100
    
    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    loss_hist = []
    
    Plot_PnL(loss_hist, net)
    Plot_Strategy(net)
    
    for epoch in range(Nepochs+1):  # loop over the dataset multiple times

        # grab a mini-batch from simulations
        PnL = SimLag1(net, mini_batch_size)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # set loss function to be conditional value at risk of 10%
        loss = -CVaR(PnL, 0.1)

        # propogate the sensitivity of the output to the model parameters 
        # backwards through the computational graph
        loss.backward()

        # update the weights and biases by taking a SGD step
        optimizer.step()

        # store running loss
        loss_hist.append(loss.item())
        
        if ((epoch) % 500 == 0 and (epoch>10) ):
            print(epoch)
            Plot_PnL(loss_hist, net, str(epoch) + " Lag 1")
            Plot_Strategy(net, str(epoch) + " Lag 1")
    
    print(epoch)
    Plot_PnL(loss_hist, net, str(Nepochs))
    Plot_Strategy(net)
    print('Finished Training')

    
    return loss_hist


def Plot_Strategy(net, epoch = "0"):
    
    t = [0,0.5*T,0.9*T]
    S = np.linspace(0.75*K2,1.25*K2,50)
    for i, time in enumerate(t):
        
        plt.subplot(1,3,i+1)
        plt.title('t=' + str(time),fontsize=16)
        
        plt.plot(S, bullspreadDelta(S, K1, K2, T-time, sigma, r), color = "skyblue")
        
        x = torch.zeros((len(S),3))
        x[:,0] = 2*time/T-1
        x[:,1] = torch.tensor(2*S/S0-1)
        x[:,2] = torch.tensor(2*S/S0-1)
        
        alpha = net(x)
        
        plt.plot(S, alpha.detach().numpy(), color = "orange")
        
        if i == 1:
            plt.xlabel('S', fontsize=16)
        if i == 0:
            plt.ylabel(r'$\Delta(S)$', fontsize=16)
        if i == 2:
            plt.legend(["Black-Scholes", "Network Estimate"])
        
        plt.ylim(0,1)
        
    plt.suptitle(r"$\Delta(S)$ of BS and ANN Strategies, Epoch " + epoch, fontsize = 16)
    plt.show()
    
    


def Plot_PnL(loss_hist, net, name="0"):
    plt.figure(figsize=(10,6))
    plt.suptitle("ANN Loss History And PnL Distribution of Terminal Strategy, Epoch: " + name, fontsize=12)
    plt.subplot(1,2,1)
    plt.plot(loss_hist, color = "red")
    plt.xlabel('iteration',fontsize=16)
    plt.ylabel('loss',fontsize=16)

    plt.subplot(1,2,2)
    PnL = SimLag1(net, 10_000)
    PnL_BS = sim_valuation(10_000)
    plt.hist(PnL_BS.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.6,color="skyblue", ec="blue",label="Black-Scholes")
    plt.hist(PnL.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.6,color="orange",ec="black",label="Network Prediction")    
    plt.xlabel('P&L',fontsize=16)
    plt.ylabel('Count',fontsize=12)
    plt.legend()
    
    plt.tight_layout(pad=2)
    plt.show()


netL1 = HedgeNet(3, 50, 3)

loss_hist_lag1 = FitNet1(netL1, 3000)

#%% plotting lag 1 with original data

fig, axes = plt.subplots()
axes.plot(loss_hist_1L, color = "red", label = r"Lag 0, $CVaR_{0.1} = 0.584$", alpha = 0.5)
axes.plot(loss_hist_lag1, color = "green", label = r"Lag 1, $CVaR_{0.1} = 0.582$", alpha = 0.5)

axes.legend(loc = 1, fontsize = 16)
axes.set_xlim(-10, 1000)
axes.set_ylim(0.5, 1)
fig.suptitle("Loss History of 2D and 3D Inputs First 1000 Epochs", fontsize = 16, y = 0.95)

#%% plotting lag 1 pnl with original data and bs

fig, axes = plt.subplots()
PnL = SimLag1(netL1, 10_000)
PnL_BS = sim_valuation(10_000)
PnLO = Sim(net, 10000)

plt.hist(PnL_BS.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.1,color="skyblue", ec="blue",label=r"BS, $CVaR_{0.1} = 0.512$")
plt.hist(PnL.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.3,color="blue",ec="black",label=r"Lag 0, $CVaR_{0.1} = 0.5766$")
plt.hist(PnLO.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.3,color="green",ec="blue",label=r"Lag 1, $CVaR_{0.1} = 0.5759$")

plt.xlim(-0.8, 0)
plt.xlabel('P&L',fontsize=16)
plt.ylabel('Count',fontsize=12)
plt.title("PnL of 2D and 3D Inputs ANN Strategy", fontsize = 16)
plt.legend(fontsize = 16)
