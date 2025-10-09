import torch
from torch.nn import CosineSimilarity, Module, MSELoss, L1Loss, Softmax, LogSoftmax

# MOCHA's L_dist (Methodology - Training Objectives and Strategy)
class ReconLoss(Module):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss()
        self.l1 = L1Loss()

    def forward(self, x, y):
        return self.mse(x,y) + self.l1(x, y)

# See AuXFT codebase for more details
class CosineLoss(Module):
    def __init__(self):
        super().__init__()
        self.sim = CosineSimilarity()

    def forward(self, x, y):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        l = 1. - self.sim(x, y)
        return l.mean()

# MOCHA's L_emb (Methodology - Training Objectives and Strategy)
class EmbeddingLoss(Module):
    def __init__(self, T=1.0):
        super().__init__()
        self.T = T
        self.softmax = Softmax(dim=-1)
        self.logsoftmax = LogSoftmax(dim=-1)

    def forward(self, x, y):
        y = y.detach()

        dxx = torch.cdist(x, x) / x.shape[-1] # normalized cross-vector distances by dimension
        dxx = dxx.masked_select(~torch.eye(x.shape[0],
                                           device=x.device,
                                           dtype=torch.bool)
                                ).view(x.shape[0], x.shape[0]-1) # remove diagonal elements

        dyy = torch.cdist(y, y) / y.shape[-1] # normalized cross-vector distances by dimension
        dyy = dyy.masked_select(~torch.eye(y.shape[0],
                                           device=y.device,
                                           dtype=torch.bool)
                                ).view(y.shape[0], y.shape[0]-1) # remove diagonal elements

        pxx = self.logsoftmax(-dxx / self.T)
        pyy = self.softmax(-dyy / self.T)

        ent = -(pyy*pxx).sum(dim=-1)
        return ent.mean()

# Running this code will generate the toy examples in Figure 3 of the paper.
if __name__ == '__main__':
    from torch.optim import SGD
    from matplotlib import pyplot as plt

    torch.manual_seed(42)
    N = 1000

    a = torch.randn(10, 3, device='cuda')
    b = torch.randn(10, 2, requires_grad=True, device='cuda')
    acc5 = []
    acc3 = []
    acc1 = []

    adist = torch.cdist(a, a)
    tka = adist.topk(k=6, largest=False)[1][:, 1:]

    loss_fn = EmbeddingLoss(T=1.0)
    optimizer = SGD([b], lr=1.)

    fig1 = plt.figure(1, figsize=(5,5))
    fig2 = plt.figure(2, figsize=(5,5))
    fig3 = plt.figure(3, figsize=(5,5))
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot(projection="3d")
    ax3 = fig3.add_subplot()
    ax1.scatter(b.detach().cpu()[:, 0], b.detach().cpu()[:, 1], marker='*', c=[f"C{c}" for c in range(10)], alpha=1, zorder=10)
    for i in range(N):
        optimizer.zero_grad()
        loss = loss_fn(b, a)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bdist = torch.cdist(b, b)
            tkb = bdist.topk(k=6, largest=False)[1][:, 1:]
            a5 = 1.*(tkb == tka)
            a3 = 1.*(tkb[:,:4] == tka[:,:4])
            a1 = 1.*(tkb[:,:2] == tka[:,:2])
            acc5.append(100*a5.mean().item())
            acc3.append(100*a3.mean().item())
            acc1.append(100*a1.mean().item())

        if i % 10 == 9:
            print(i, loss.item())
            ax1.scatter(b.detach().cpu()[:, 0], b.detach().cpu()[:, 1], c=[f"C{c}" for c in range(10)], alpha=i/N)

    ax1.set_aspect('equal')
    ax1.grid(True, 'both', 'both')
    ax2.scatter(a.cpu()[:, 0], a.cpu()[:, 1], a.cpu()[:, 2], c=[f"C{c}" for c in range(10)])
    ax2.view_init(elev=35, azim=-10, roll=0)
    ax3.plot(acc1, c="C0", alpha=.2)
    ax3.plot(acc3, c="C1", alpha=.2)
    ax3.plot(acc5, c="C2", alpha=.2)
    sacc5 = []
    sacc3 = []
    sacc1 = []
    for i, (a1, a3, a5) in enumerate(zip(acc1, acc3, acc5)):
        if i == 0:
            sacc1.append(a1)
            sacc3.append(a3)
            sacc5.append(a5)
        else:
            sacc1.append(.15*a1 + .85*sacc1[i-1])
            sacc3.append(.15*a3 + .85*sacc3[i-1])
            sacc5.append(.15*a5 + .85*sacc5[i-1])
    ax3.plot(sacc1, c="C0", label="Top1")
    ax3.plot(sacc3, c="C1", label="Top3")
    ax3.plot(sacc5, c="C2", label="Top5")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Top-K Neighbors Match Rate %")
    ax3.grid(True, 'both', 'both')
    ax3.legend()
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig1.savefig('2d.pdf')
    fig2.savefig('3d.pdf')
    fig3.savefig('acc.pdf')
    # plt.show()
