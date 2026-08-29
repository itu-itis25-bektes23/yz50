import random
from gorev4 import Value


class Neuron:
    def __init__(self, nin):
      self.b = Value( random.uniform(-1,1))
      self.w = []
      for i in range(nin):
        self.w.append(Value(random.uniform(-1,1)))
    def __call__(self,x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi*xi
        return act.tanh()

    def parameters(self): # Pyhtona ozgu metodlar icinmic cift _, __X__ 
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
      self.nout = nin
      self.w = []
      for i in range(nin):
        self.w.append(Neuron)

    def __call__(self, x):
        cal = call(Neuron)
        stop if length (Neuron) = 1
    def parameters(self):
        return self.parameters(Neuron)
