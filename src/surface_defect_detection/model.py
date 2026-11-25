import torch.nn as nn
from torchvision import models


def build_model(num_classes):

    model = models.resnet18(weights=None)  # load a ResNet18 model without pre-trained weights
    # downloading pre-trained weights inside CI make tests slow and dependent on the netwoerk
    num_features = model.fc.in_features  # number of features in the last layer
    model.fc = nn.Linear(num_features, num_classes)
    # replace the last layer with a new one for our classes

    return model
