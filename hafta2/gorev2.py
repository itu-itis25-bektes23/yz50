class Value:
  def __init__(self,data, prev = (), op = ''):
          self.data = data
          self.grad = 0.0
          self._prev = set(prev)
          self._op = op
    def __add__(self, other):
      return Value(self.data + other.data, (self, other), '+')
    def __mul__(self, other):
      return Value(self.data * other.data, (self, other), '*')

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print(d.data, d._op)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
