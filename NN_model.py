import torch
import torch.nn as nn
from torch.nn.functional import normalize

class Encoder(nn.Module):
    def __init__(self, backbone, feature_dim):
        super(Encoder, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(

            nn.Linear(self.backbone.get_sentence_embedding_dimension(),
                      self.backbone.get_sentence_embedding_dimension()),

            nn.ReLU(),
            nn.Linear(self.backbone.get_sentence_embedding_dimension(), self.feature_dim),
        )

    def forward(self, X):
        h = self.backbone.encode(X, batch_size=len(X), convert_to_numpy=False, convert_to_tensor=True)
        z = normalize(self.instance_projector(h), dim=1)
        return z

    def forward_text(self,X):
        h = self.backbone.encode(X, batch_size=len(X), convert_to_numpy=False, convert_to_tensor=True)
        if h.dim()==1:
            h = h.unsqueeze(0)
        z = self.instance_projector(h)
        z = normalize(z, dim=1)
        z = torch.flatten(z, start_dim=1)
        return z

