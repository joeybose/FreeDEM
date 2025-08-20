import abc
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseNoiseSchedule(ABC):
    def __call__(self, x):
        outf = self.tt(x)
        doutf = self.dtt(x)
        
        return outf, doutf
    
    @abstractmethod
    def g(t):
        # Returns g(t)
        pass

    @abstractmethod
    def h(t):
        # Returns \int_0^t g(t)^2 dt
        pass

    @abc.abstractmethod
    def tt(self, x): 
        pass

    @abc.abstractmethod
    def dtt(self, x): 
        pass


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def g(self, t):
        return torch.full_like(t, self.beta**0.5)

    def h(self, t):
        return self.beta * t

    def tt(self, t):
        out = t 
        return out

    def dtt(self, t):
        out = torch.tensor(1.0).to(self.device)
        return out

class QuadraticNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.sqrt(self.beta * 2 * t)

    def h(self, t):
        return self.beta * t**2


class PowerNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta, power):
        self.beta = beta
        self.power = power

    def g(self, t):
        return torch.sqrt(self.beta * self.power * (t ** (self.power - 1)))

    def h(self, t):
        return self.beta * (t**self.power)


class SubLinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.sqrt(self.beta * 0.5 * 1 / (t**0.5 + 1e-3))

    def h(self, t):
        return self.beta * t**0.5


class GeometricNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = self.sigma_max / self.sigma_min

    def g(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then g(t) = sigma_min * sigma_d^t * sqrt{2 * log(sigma_d)}
        # See Eq 192 in https://arxiv.org/pdf/2206.00364.pdf
        return self.sigma_min * (self.sigma_diff**t) * ((2 * np.log(self.sigma_diff)) ** 0.5)

    def h(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then h(t) = \int_0^t g(z)^2 dz = sigma_min * sqrt{sigma_d^{2t} - 1}
        # see Eq 199 in https://arxiv.org/pdf/2206.00364.pdf
        return (self.sigma_min * (((self.sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2
