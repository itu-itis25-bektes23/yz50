import math
class Value:
    def __init__(self, data, prev=(), op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(prev)
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data  * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

// Test degerlerini ekliyorum:

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(-2.0)

e = a * b
d = e + c
L = d * f

L.backward()
print(a.grad, b.grad, c.grad, d.grad, e.grad, f.grad)
