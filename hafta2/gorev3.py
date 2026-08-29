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


    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

# Test degerlerini ekliyorum: // ile ekleyemiyormusuz.
 
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(-2.0)

e = a * b
d = e + c
L = d * f

L.backward()
print(a.grad, b.grad, c.grad, d.grad, e.grad, f.grad)


x1 = Value(2.0);  x2 = Value(0.0)
w1 = Value(-3.0); w2 = Value(1.0)
bias  = Value(6.8813735870195432)

n = x1*w1 + x2*w2 + bias
o = n.tanh()
o.backward()

print(x1.grad, w1.grad, x2.grad, w2.grad)
