import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.container import Sequential
dropout = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #Input block
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0, bias=False) 
        self.gn1=nn.GroupNorm(4, 16)
        self.ln1=nn.LayerNorm(26)
        self.bn1=nn.BatchNorm2d(16)
        self.drop1=nn.Dropout(dropout)  
        #Output - 26*26
        
        
        #Convolution Block
        self.conv2 = nn.Conv2d(16, 20, 3, padding=0, bias=False)
        self.bn2=nn.BatchNorm2d(20)
        self.gn2=nn.GroupNorm(5, 20)
        self.ln2=nn.LayerNorm(24)
        self.drop2=nn.Dropout(dropout)              
         #Output - 24*24
        
        self.conv3 = nn.Conv2d(20, 10, 3, padding=0, bias=False)
        self.gn3=nn.GroupNorm(2, 10)
        self.ln3=nn.LayerNorm(22)
        self.bn3=nn.BatchNorm2d(10)
        self.drop3=nn.Dropout(dropout)                
         #Output - 22*22
        
        #Transaction Block
        self.pool1 = nn.MaxPool2d(2, 2)   #Output - 11*11
        self.conv4 = nn.Conv2d(10, 20, 1, padding=0, bias=False)
        self.gn4=nn.GroupNorm(5, 20)
        self.ln4=nn.LayerNorm(11)
        self.bn4=nn.BatchNorm2d(20)
        self.drop4=nn.Dropout(dropout)                
         #Output - 11*11
        
        #Convolution Block
        self.conv5=nn.Conv2d(20, 10, 3, padding=0, bias=False)
        self.gn5=nn.GroupNorm(2, 10)
        self.ln5=nn.LayerNorm(9)
        self.bn5=nn.BatchNorm2d(10)
        self.drop5=nn.Dropout(dropout)                
         #Output - 9*9
        
        self.conv6 = nn.Conv2d(10, 16, 3, padding=0, bias=False)
        self.gn6=nn.GroupNorm(2, 16)
        self.ln6=nn.LayerNorm(7)
        self.bn6=nn.BatchNorm2d(16)
        self.drop6=nn.Dropout(dropout)                
         #Output - 7*7
        
        #Output block
        self.conv7 = nn.Conv2d(16, 8, 3, padding=0, bias=False)
        self.gn7=nn.GroupNorm(2, 8)
        self.bn7=nn.BatchNorm2d(8)
        self.ln7=nn.LayerNorm(5)
        self.drop7=nn.Dropout(dropout)                
         #Output - 5*5
        self.conv8 = nn.Sequential(
            nn.Conv2d(8, 10, 1, padding=0, bias=False),  
        ) #Output - 5*5
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) #Output - 1*1
        #self.ReLU=nn.ReLU()


    def forward(self, x, type_of_norm):
        type_of_norm = type_of_norm.lower()
        
        if type_of_norm == 'gn':
            x = self.conv1(x)
            x = self.gn1(x)
            x = F.relu(x)
            x = self.drop1(x)
            
            x = self.conv2(x)
            x = self.gn2(x)
            x = F.relu(x)
            x = self.drop2(x)
            
            x = self.conv3(x)
            x = self.gn3(x)
            x = F.relu(x)
            x = self.drop3(x)
            
            x = self.pool1(x)
            
            x = self.conv4(x)
            x = self.gn4(x)
            x = F.relu(x)
            x = self.drop4(x)
            
            x = self.conv5(x)
            x = self.gn5(x)
            x = F.relu(x)
            x = self.drop5(x)
            
            x = self.conv6(x)
            x = self.gn6(x)
            x = F.relu(x)
            x = self.drop6(x)
            
            x = self.conv7(x)
            x = self.gn7(x)
            x = F.relu(x)
            x = self.drop7(x)
            
        if type_of_norm == 'ln':
            x = self.conv1(x)
            x = self.ln1(x)
            x = F.relu(x)
            x = self.drop1(x)

            x = self.conv2(x)
            x = self.ln2(x)
            x = F.relu(x)
            x = self.drop2(x)

            x = self.conv3(x)
            x = self.ln3(x)
            x = F.relu(x)
            x = self.drop3(x)

            x = self.pool1(x)

            x = self.conv4(x)
            x = self.ln4(x)
            x = F.relu(x)
            x = self.drop4(x)

            x = self.conv5(x)
            x = self.ln5(x)
            x = F.relu(x)
            x = self.drop5(x)

            x = self.conv6(x)
            x = self.ln6(x)
            x = F.relu(x)
            x = self.drop6(x)

            x = self.conv7(x)
            x = self.ln7(x)
            x = F.relu(x)
            x = self.drop7(x)

        if type_of_norm == 'bn':
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.drop1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.drop2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.drop3(x)

            x = self.pool1(x)

            x = self.conv4(x)
            x = self.bn4(x)
            x = F.relu(x)
            x = self.drop4(x)

            x = self.conv5(x)
            x = self.bn5(x)
            x = F.relu(x)
            x = self.drop5(x)

            x = self.conv6(x)
            x = self.bn6(x)
            x = F.relu(x)
            x = self.drop6(x)

            x = self.conv7(x)
            x = self.bn7(x)
            x = F.relu(x)
            x = self.drop7(x)
        
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
