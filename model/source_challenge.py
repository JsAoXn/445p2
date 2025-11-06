import torch
import torch.nn as nn
from math import sqrt
from utils import set_random_seed

__all__ = ["Source"]

class Source_Challenge(nn.Module):
    def __init__(self, num_classes=8) -> None: # Pass in num_classes
        """Define improved model architecture."""
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        

        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       
       
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(p=0.5) 
        
       
        self.fc1 = nn.Linear(128, num_classes) 
     
    
        
   
  

    

      
        

        
       
       



 


    
        
       
        
       

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
 
        nn.init.normal_(self.fc1.weight, mean=0.0, std=sqrt(1.0/self.fc1.in_features))
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool(x)     
        x = torch.flatten(x, 1) 
        x = self.dropout(x)     
        x = self.fc1(x)         
        return x