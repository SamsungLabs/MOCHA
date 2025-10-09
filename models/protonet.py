import torch
from torch import nn
from torch.nn import functional as F

# See AuXFT codebase for more details
def dcos(x, y):
    d = 1 - F.cosine_similarity(x, y, dim=0) # pylint: disable=not-callable
    return d

# See AuXFT codebase for more details
def dl1(x, y):
    d = F.l1_loss(x, y)
    return d

# See AuXFT codebase for more details
def dl2(x, y):
    d = F.mse_loss(x, y)
    return d

# Prototypical classifier used by MOCHA
# See AuXFT codebase for more details
class Conditional(nn.Module):
    def __init__(self,
                 consider_coarse=True,
                 norm=False):
        super().__init__()

        self.consider_coarse = consider_coarse
        self.norm = norm

        self.prototypes = {}
        self.counts = {}

        self.dists = [dcos, dl1, dl2]

    def reset(self):
        self.prototypes = {}
        self.counts = {}

    def train_protos(self, x, y):
        for i, sample in enumerate(x):
            for f, _, cl in sample:
                if self.norm:
                    f = f / f.norm()
                fl = y[i].item()
                if not cl in self.prototypes:
                    self.prototypes[cl] = {-1: torch.zeros_like(f)} if self.consider_coarse else {}
                    self.counts[cl] = {-1: 0} if self.consider_coarse else {}
                if not fl in self.prototypes[cl]:
                    self.prototypes[cl][fl] = torch.zeros_like(f)
                    self.counts[cl][fl] = 0

                if self.consider_coarse:
                    self.prototypes[cl][-1] = (self.prototypes[cl][-1]*self.counts[cl][-1] + f)/(self.counts[cl][-1]+1)
                    self.counts[cl][-1] += 1

                self.prototypes[cl][fl] = (self.prototypes[cl][fl]*self.counts[cl][fl] + f)/(self.counts[cl][fl]+1)
                self.counts[cl][fl] += 1

    def forward(self, x, preds):
        for xs, ps in zip(x, preds):
            for (f, _, cl), box in zip(xs, ps):
                if self.norm:
                    f = f / f.norm()
                if cl in self.prototypes:
                    cfs, protos = list(self.prototypes[cl]), list(self.prototypes[cl].values())
                    dists = torch.zeros(len(self.dists), len(protos), device=ps.device)
                    for di, dist in enumerate(self.dists):
                        for pi, p in enumerate(protos):
                            dists[di, pi] = dist(f,p)
                    probs = F.softmax(1./(dists+1e-5), dim=-1).mean(dim=0)
                    cf = cfs[probs.argmax()]
                    box[-1] = cf
                else:
                    box[-1] = -1
        return preds

    def get_protos(self):
        return {cl: {cf: v.numpy().tolist() for cf, v in d.items()} for cl, d in self.prototypes.items()}

# See AuXFT codebase for more details
class BaseProtonet(nn.Module):
    def __init__(self,
                 consider_coarse=True,
                 norm=False):
        super().__init__()
        self.consider_coarse = consider_coarse
        self.norm = norm

        self.prototypes = {}
        self.counts = {}

        self.dist = dl2

    def reset(self):
        self.prototypes = {}
        self.counts = {}

    def train_protos(self, x, y):
        for i, sample in enumerate(x):
            for f, _, cl in sample:
                fl = y[i].item()

                if self.norm:
                    f /= f.norm()

                if fl not in self.prototypes:
                    self.prototypes[fl] = torch.zeros_like(f)
                    self.counts[fl] = 0

                if self.consider_coarse and -cl not in self.prototypes:
                    self.prototypes[-cl] = torch.zeros_like(f)
                    self.counts[-cl] = 0

                if self.consider_coarse:
                    self.prototypes[-cl] = (self.counts[-cl]*self.prototypes[-cl] + f)/(self.counts[-cl] + 1)
                    self.counts[-cl] += 1

                self.prototypes[fl] = (self.counts[fl]*self.prototypes[fl] + f)/(self.counts[fl] + 1)
                self.counts[fl] += 1

    def forward(self, x, preds):
        fls, pts = list(self.prototypes), list(self.prototypes.values())
        ds = torch.zeros(len(pts))

        for xs, ps in zip(x, preds):
            for (f, _, _), box in zip(xs, ps):
                if self.norm:
                    f /= f.norm()

                for i, pt in enumerate(pts):
                    ds[i] = 1./(self.dist(f,pt) + 1e-5)

                box[-1] = fls[ds.argmax()]

        return preds

# See AuXFT codebase for more details
class SimpleShot(BaseProtonet):
    def __init__(self, consider_coarse=True, norm=True):
        super().__init__(consider_coarse=consider_coarse, norm=True)
