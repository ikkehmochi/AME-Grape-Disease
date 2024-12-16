import torch
import torch.nn as nn
import torchvision.models as models

class AdaptiveEnsembleModel(nn.Module):
    def __init__(self,model1, model2, num_classes=4):
        super(AdaptiveEnsembleModel, self).__init__()
        
        self.model1 = model1
        self.model2 = model2
        self.num_features = 1280
        
        self.model1.classifier = nn.Identity()
        self.model2.classifier = nn.Identity()
        self.adaptive_layer = nn.Linear(self.num_features * 2, num_classes)
        
    def forward(self, x):
        features1 = self.model1(x)
        features2 = self.model2(x)
        
        combined_features = torch.cat((features1, features2), dim=1)
        
        output = self.adaptive_layer(combined_features)
        
        return output

