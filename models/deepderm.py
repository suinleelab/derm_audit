#!/usr/bin/env python
import torch
import torchvision
from torchvision import transforms

class DeepDermClassifier(torch.nn.Module):
    '''Wrapper for DeepDerm classifier; for use with (-1,1)-scaled images.'''
    def __init__(self):
        super().__init__()
        self.image_size = 299
        self.positive_index = 1
        model_path = 'pretrained_classifiers/deepderm.pth'

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # load model
        self.model = torchvision.models.inception_v3(init_weights=False)
        self.model.fc = torch.nn.Linear(2048, 2)
        self.model.AuxLogits.fc = torch.nn.Linear(768, 2)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model._ddi_name = 'DeepDerm' 
        self.model._ddi_threshold = 0.687
        self.model._ddi_web_path = 'https://drive.google.com/uc?id=1OLt11htu9bMPgsE33vZuDiU5Xe4UqKVJ'
        self.eval()

    def forward(self, x):
        '''Expects an image scaled to (-1,1) range. Rescale to the classifer's
        native range, then pass through the classifier and return the softmax of
        the classifier's output'''
        # rescale to (0,1) range
        rescaled = x*0.5+0.5
        # rescale to classifier's native range
        rescaled = self.normalize(rescaled)
        # call the classifier
        return torch.nn.functional.softmax(self.model(rescaled), dim=1)
