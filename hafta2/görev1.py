class Value:
  def __init__(self,data, prev = (), op = ''):
          self.data = data
          self.grad = 0.0
          self._prev = set(prev)
          self._op = op
    def __add__(self, other):
        return Value( data  , op , prev )
