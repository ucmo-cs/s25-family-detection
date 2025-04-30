import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        #use the pretrained resnet18 as base feature extractor
        #use resnet18, model thats pretrained on imagenet since it already gets low-lwevel features like edges, shapes, skin tones
        base_model = models.resnet18(pretrained=True)
        #remove classification head
        #get rid of classification layer since we dont really gaf about classifying objects (only features from face)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        ##maps high-dimensional feature vector(512) to a clean 128D vector (keep compact and alligned with facenet conventions, people way smarter than me)
        self.embedding = nn.Linear(base_model.fc.in_features, embedding_dim)

    def forward_once(self, x):
        #ouput shape (batch_size, 512, 1, 1)
        x = self.encoder(x)
        #flatten
        x = x.view(x.size(0), -1)
        #down to 128d
        x = self.embedding(x)
        #normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        #label = 1 for kin, 0 for non kin
        distances = F.pairwise_distance(output1, output2)
        loss = label * distances.pow(2) + \
               (1 - label) * F.relu(self.margin - distances).pow(2)
        return loss.mean()