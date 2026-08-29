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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
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


  def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward

        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')

        def _backward():
            self.grad += k* self.data**(k-1) * out.grad    
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * Value(-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

a = Value(3.0)
b = a ** 2
b.backward()
print(a.grad)       

c = Value(0.0)
d = c.exp()
d.backward()
print(c.grad)      


x = Value(0.8814)
o1 = x.tanh()
o1.backward()
g1 = x.grad

x = Value(0.8814)         
e2 = (2*x).exp()
o2 = (e2 - 1) / (e2 + 1)
o2.backward()
g2 = x.grad

print(o1.data, o2.data)
print(g1, g2)
