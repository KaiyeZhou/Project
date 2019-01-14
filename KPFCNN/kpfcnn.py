import torch
import torch.nn as nn

class kpfcnn(nn.module):
    def __init__(self, num_input):
        super(kpfcnn, self).__init__()
        self.conv1 = nn.Conv2d(num_input, tongdaoshu, 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(tongdaoshu, 1024)
        self.fc2 = nn.Linear(1024, 4)
        self.m = nn.Softmax()
    
    def forward(self, x):
        c1 = self.relu(self.conv1(x))
        f1 = self.fc1(c1)
        f2 = self.fc2(f1)
        out = self.m(f2)

        return out