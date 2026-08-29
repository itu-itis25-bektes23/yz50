import random
from gorev4 import Value


class Neuron:
    def __init__(self, nin):
      self.b = Value( random.uniform(-1,1))
      self.w = []
      for i in range (nin)
        self.w.append(Value(random.uniform(-1,1)))

      
