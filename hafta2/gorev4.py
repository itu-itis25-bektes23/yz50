    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += x * out.grad 
        out._backward = _backward

        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')

        def _backward():
            self.grad += k* x**(k-1) * out.grad    
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * Value(-1.0)

    def __sub__(self, other):
        return self + (-other)
