from torch import nn
import torch
import torch.nn.functional as Func


class cosine_similarity_fun(nn.Module):
    def __init__(self):
        super(cosine_similarity_fun, self).__init__()

    def forward(self, text):
        text_0 = text[:int(text.size(0) / 2), :]  # [32, 2048]
        text_1 = text[int(text.size(0) / 2):, :]  # [32, 2048]
        loss = 1 - Func.cosine_similarity(text_0, text_1, dim=1).mean()
        return loss

class ConsistencyCos(nn.Module):
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1) # [64, 2048]

        feat_0 = feat[:int(feat.size(0)/2),:] # [32, 2048]
        feat_1 = feat[int(feat.size(0)/2):,:] # [32, 2048]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1) # [32, 1]
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False) # [32, 1]
        if torch.cuda.is_available():
            labels = labels.cuda()
        loss = self.mse_fn(cos, labels)
        return loss

class ConsistencyL2(nn.Module):
    def __init__(self):
        super(ConsistencyL2, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        loss = self.mse_fn(feat_0, feat_1)
        return loss

class ConsistencyL1(nn.Module):
    def __init__(self):
        super(ConsistencyL1, self).__init__()
        self.L1_fn = nn.L1Loss()

    def forward(self, feat):
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        loss = self.L1_fn(feat_0, feat_1)
        return loss

if __name__ == '__main__':
    t = torch.randn([64, 128, 74, 74])
    cos = cosine_similarity_fun()
    loss = cos(t)
    print(loss)