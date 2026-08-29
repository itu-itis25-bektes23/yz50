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
      self.neurons = []
      for i in range(nout):
        self.neurons.append(Neuron(nin))

    def __call__(self, x):
        outs = []
        for n in self.neurons:
            outs.append(n(x))
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params
