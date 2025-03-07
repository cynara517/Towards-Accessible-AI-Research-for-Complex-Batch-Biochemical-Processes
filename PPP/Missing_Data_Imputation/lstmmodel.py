import torch
import torch.nn as nn

from torch.autograd import Variable

# 定义LSTM模型
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 10

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.9)
        #self.linear1=nn.Linear(hidden_size,hidden_size-1)
        self.relu=nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
#x.size(0)为0维度的数据数量
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        #ula, (h_out, _) = self.lstm(x, (h_0, c_0))
#使用最后一个h——out进行预测输出
        h_out = h_out[-1,:].view(-1, self.hidden_size)
        #out=self.linear1(h_out)
        out=self.relu(h_out)
        out=self.fc(h_out)
        #out=self.relu(h_out)
        #out = self.fc(h_out)

        return out
# 初始化模型参数
num_classes=1
num_layers=2
input_size=12
hidden_size=16
model=LSTM(num_classes,input_size,hidden_size,num_layers)