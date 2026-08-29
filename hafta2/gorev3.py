class Value:
  def __init__(self,data, prev = (), op = ''):
          self.data = data
          self.grad = 0.0
          self._prev = set(prev)
          self._op = op
          self._backward = lambda: None
    def __add__(self, other):
      out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += self.data * out.grad
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
      out = Value(self.data + other.data, (self, other), '+')
    def __mul__(self, other):
        def _backward():
            self.grad += other.grad * self.grad
            other.grad += other.grad + self.grad
        out._backward = _backward
        return out
      return Value(self.data * other.data, (self, other), '*')

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print(d.data, d._op)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c
f = Value(-2.0)
L = d * f

print(L)
