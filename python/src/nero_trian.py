import numpy as np
import neurolab as nl
input=[[1.0,0.0],[0.0,1.0],[1.0,1.0],[0.0,0.0]]
target = [[0.0],[0.0],[1.0],[0.0]]
net = nl.net.newff([[0, 1], [0, 1]], [5, 1])
net.error=nl.net.error.MSE()
err = net.train(input, target)
print(err)