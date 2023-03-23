import torch
from torch import nn
import time

def set_bn_online(model):
    set = False
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and "stem" in n:
            m.running_mean = None
            m.running_var = None
            set = True

    if not set:
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and not set:
                m.running_mean = None
                m.running_var = None
                set = True
                print(f"set {n} online!")
                return
            
def add_kdomain(model, momentums=[0.4, 0.01]):
    old = model.backbone.stem.bn
    v = BatchNormAdaptKDomain(old.num_features, device=torch.device("cuda")).to(old.running_var.device)
    v.running_var = old.running_var
    v.running_mean = old.running_mean
    v.eps = old.eps
    v.momentum = old.momentum
    v.affine = old.affine
    v.weight = old.weight
    v.bias = old.bias
    v.momentums = momentums
    v.eval()
    model.backbone.stem.bn = v

class BatchNormAdaptKDomain(nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        self.num_domains = 2
        self.initialized = [False for i in range(self.num_domains)]

        # keep a different copy to the running mean / var and copy back
        # self.means = torch.zeros(self.num_domains, num_features, dtype=torch.float32, requires_grad=False).to(device)
        # self.vars = torch.ones(self.num_domains, num_features, dtype=torch.float32, requires_grad=False).to(device)

        self.register_buffer("means", torch.zeros(self.num_domains, num_features, dtype=torch.float32, requires_grad=False).to(device))
        self.register_buffer("vars", torch.ones(self.num_domains, num_features, dtype=torch.float32, requires_grad=False).to(device))

        self.register_buffer("current_mean", torch.zeros(num_features, dtype=torch.float32, requires_grad=False).to(device))
        self.register_buffer("current_var", torch.ones(num_features, dtype=torch.float32, requires_grad=False).to(device))

        self.calculator = nn.BatchNorm2d(self.num_features)

        self.o_running_mean = None
        self.o_running_var = None

        self.pred = 0.0
        self.momentums = [0.99, 0.5] # original domain should slowly update

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b = input.shape[0]
        
        # just use for calculating forward mean, variance
        # magnitudes of order faster than torch.mean()
        torch.batch_norm(input, weight=None, bias=None, running_mean=self.current_mean, running_var=self.current_var, training=True, momentum=1.0, eps=self.eps, cudnn_enabled=True)

        if self.o_running_mean is None:
            self.o_running_mean = self.running_mean.detach().clone()
            self.o_running_var = self.running_var.detach().clone()

            # copy running for 1st domain
            self.means[0] = self.o_running_mean
            self.vars[0] = self.o_running_var

            # copy current for 2nd domain
            self.means[1] = self.current_mean
            self.vars[1] = self.current_var

        d = torch.mean(torch.pow(self.current_mean.unsqueeze(0).repeat((self.num_domains, 1)) - self.means, 2.0), dim=1)

        idx = torch.argmin(d).item()

        self.pred = idx

        self.means[idx] = self.means[idx] * self.momentums[idx] + (1.0 - self.momentums[idx]) * self.current_mean
        self.vars[idx] = self.vars[idx] * self.momentums[idx] + (1.0 - self.momentums[idx]) * self.current_var

        # clone for BatchNorm2d
        self.running_mean = self.means[idx] #.detach().clone()
        self.running_var = self.vars[idx] #.detach().clone()
        
        return torch.batch_norm(input, self.weight, self.bias, self.running_mean, \
                                    self.running_var, training=False, momentum=0, \
                                        eps=self.eps, cudnn_enabled=True)
        # return super().forward(input)
    
if __name__ == "__main__":
    device = torch.device("cuda")
    m = BatchNormAdaptKDomain(64).to(device).train()

    images = torch.randn((1, 64, 768, 1280)).to(device)

    o = m(images)

    print(torch.mean(o, dim=(0, 2, 3)), torch.var(o, dim=(0, 2, 3)))

    # speed test
    num = 1000

    start = time.time()
    for i in range(num):
        m(images)
    end = time.time()

    ave_duration = (end - start) / num

    print("ave_duration", ave_duration)