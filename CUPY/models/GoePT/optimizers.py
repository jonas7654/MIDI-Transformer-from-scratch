import math
import os
from typing import List, Union

if not os.getenv('GOEPT_CPU', False):
    import cupy as xp
    import numpy as np
else:
    import numpy as xp
    np = xp


class Adam:
    def __init__(self,
                    params: List[xp.ndarray],
                    lr: float=1e-3,
                    lr_min: Union[float, None] = None,
                    warmup_iters: int=16,
                    decay_iters: int=1024,
                    betas: tuple[float, float]=(0.9, 0.99),
                    weight_decay_rates: Union[List[float], None] = None,
                    eps: float=1e-8) -> None:

        if isinstance(lr_min, type(None)):
            lr_min = lr/10

        self.lr = lr
        self.lr_min = lr_min

        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.betas = betas
        self.eps = eps

        if not isinstance(weight_decay_rates, type(None)):
            self.weight_decay_rates = weight_decay_rates
        else:
            self.weight_decay_rates = [0]*len(params)

        self.m = [xp.zeros_like(param) for param in params]
        self.v = [xp.zeros_like(param) for param in params]

        self.t = 0


    def step(self, params_in: List[xp.ndarray],
                        grads: List[xp.ndarray]) -> List[xp.ndarray]:

        self.t += 1

        params_out = []

        for i, (param, grad) in enumerate(zip(params_in, grads)):

            self.m[i] = self.betas[0]*self.m[i] + (1 - self.betas[0])*grad
            self.v[i] = self.betas[1]*self.v[i] + (1 - self.betas[1])*grad**2

            m_hat = self.m[i]/(1 - self.betas[0]**self.t)
            v_hat = self.v[i]/(1 - self.betas[1]**self.t)

            params_out.append(param - self.get_lr()*(m_hat/(np.sqrt(v_hat) + self.eps) + self.weight_decay_rates[i]))

        return params_out
    

    def get_lr(self):
        '''
        learning rate decay scheduler (cosine with warmup)
        '''

        # 1) linear warmup for warmup_iters steps
        if self.t < self.warmup_iters:
            return self.lr*(self.t + 1)/(self.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if self.t > self.decay_iters:
            return self.lr_min
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.t - self.warmup_iters)/(self.decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5*(1.0 + math.cos(math.pi*decay_ratio)) # coeff ranges 0..1
        return self.lr_min + coeff*(self.lr - self.lr_min)
