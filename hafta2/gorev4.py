    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += x * out.grad 
        out._backward = _backward

        return out

  
