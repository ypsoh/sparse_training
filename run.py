import torch

class FeedForward(torch.nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            layer_type=0, # 0: nn.Linear, 1: XsmmLinear(Dense), 2: XsmmLinear(Sparse)
            last_layer=False,
            use_sparse_kernels=False):

        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if layer_type == 0:
            self.fc = torch.nn.Linear(self.input_size, self.hidden_size)

        """
        if use_sparse_kernels:
            self.fc1 = pcl_mlp.XsmmLinear(input_size, hidden_size)
        else:
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        """

        if last_layer:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        hidden = self.fc(x)
        output = self.activation(hidden)
        return output 

class LinearNet():
    def __init__(
            self,
            input_size=1024,
            hidden_size=512,
            num_layers=3,
            device=0,
            layer_type=0):

        # device gpu or cpu
        self.device = device

        # Define linear_layer_list and add first layer
        linear_layer_list = [FeedForward(input_size, hidden_size)]

        for n in range(num_layers-1):
            linear_layer_list.append(FeedForward(hidden_size, hidden_size))

        linear_layer_list.append(FeedForward(hidden_size, 1, last_layer=True))

        # define linear layers as module list
        self.model = torch.nn.Sequential(*torch.nn.ModuleList(linear_layer_list))
        return None

import numpy
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import time

class MockDataset():
    def __init__(self, n_samples=320, n_features=256, cluster_std=1.5):
        self.x, self.y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=2,
                cluster_std=cluster_std,
                shuffle=True)

        self.train_test_split()

    def train_test_split(self):
        x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=0.2, random_state=42)

        # Assign labels
        def blob_label(y, label, loc):
            target = numpy.copy(y)
            for l in loc:
                target[y==l]=label
            return target

        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
        y_train = torch.FloatTensor(blob_label(y_train, 1, [1]))

        x_test = torch.FloatTensor(x_test)
        y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
        y_test = torch.FloatTensor(blob_label(y_test, 1, [1]))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

def run_training(N, C, K):
    lnet = LinearNet(input_size=C, hidden_size=K)
    model = lnet.model
    md = MockDataset(n_samples=N, n_features=C)

    x_train = md.x_train
    y_train = md.y_train
    x_test = md.x_test
    y_test = md.y_test

    # Define training settings
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    model.train()
    epoch = 100

    t_fp = 0.0
    t_bp = 0.0
    for epoch in range(epoch):
        optimizer.zero_grad()

        t_fp_start = time.time()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        t_fp_end = time.time()

        loss = criterion(y_pred.squeeze(), y_train)
                           
        # print('Epoch {}: train loss: {}, duration: {}'.format(epoch, loss.item(), te_epoch - ts_epoch))

        # Backward pass
        t_bp_start = time.time()
        loss.backward()
        optimizer.step()
        t_bp_end = time.time()

        t_fp += t_fp_end - t_fp_start
        t_bp += t_bp_end - t_bp_start


    print("N: {}, C: {}, K:{}".format(N, C, K))
    print("Average FP time: {}".format(t_fp/epoch))
    print("Average BP time: {}".format(t_bp/epoch))
    print()

run_training(320, 128, 128)
run_training(320, 256, 256)
run_training(320, 512, 512)
run_training(320, 1024, 1024)
run_training(320, 2048, 2048)
