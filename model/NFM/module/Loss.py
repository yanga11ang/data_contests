import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MymseLoss(nn.Module):

    def __init__(self,):
        super(MymseLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def forward(self, q_p, q_o):
        # (batch,56)
        # 拆前后
        return self.loss_fn(q_p,q_o)

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

if __name__ =="__main__":
    loss = MymseLoss()
    q_p = np.array([[0,0,0,0]])
    q_o = np.array([[0,3,0,2]])
    q_p = torch.tensor(q_p,dtype=torch.float)
    q_o = torch.tensor(q_o, dtype=torch.float)
    print(loss.predict(q_p,q_o))