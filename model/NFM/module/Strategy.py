import torch
import torch.nn as nn
from ..BaseModule import BaseModule
class Plain(BaseModule):
    def __init__(self,model = None,
                 train_loss = None,
                 valid_loss=None ,
                 ):
        super(Plain, self).__init__()
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss

    def forward(self,data):
        src = torch.einsum("ijk->jik",[data['feature']])
        results = self.model(src)
        # print(results.shape)
        # results = torch.einsum("ijk->jik",[results])
        # results = results.squeeze(2)
        target = data['target'].squeeze(2)
        return self.train_loss(results, target) # 预测， 结果

    def get_valid_loss(self,data):
        src = torch.einsum("ijk->jik", [data['feature']])
        results = self.model(src)
        # results = torch.einsum("ijk->jik", [results])
        # results = results.squeeze(2)
        target = data['target'].squeeze(2)
        return self.valid_loss(results, target).item()  # 预测， 结果

    def predict(self,data):
        src = torch.einsum("ijk->jik", [data['feature']])
        results = self.model(src)
        # results = torch.einsum("ijk->jik", [results])
        # results = results.squeeze(2)
        target = data['target'].squeeze(2)
        return results.cpu().data.numpy(),target.cpu().data.numpy()