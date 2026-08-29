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
