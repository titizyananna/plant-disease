import torch.nn as nn
import timm
import torch
import numpy as np

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None   # reverse + scale

class GRL(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, lambda_): return GradientReversalFn.apply(x, lambda_)

def get_lambda(epoch, total_epochs):
    p = epoch / total_epochs
    return (2 / (1 + np.exp(-5 * p)) - 1)

class DiseaseClassifier(nn.Module):
    def __init__(self, num_diseases, num_plants):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features

        for p in self.backbone.parameters():
            p.requires_grad = False

        for block in self.backbone.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        #Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_diseases),
        )

        #Plant classification head
        self.grl = GRL()
        self.domain_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_plants),
        )

    def forward(self, x, lambda_=1.0):
        feat = self.backbone(x)
        feat = self.projector(feat)

        #Disease prediction
        disease_logits = self.disease_head(feat)

        #Domain adversarial prediction (gradient reversed)
        feat_rev      = self.grl(feat, lambda_)
        domain_logits = self.domain_head(feat_rev)

        return disease_logits, domain_logits
