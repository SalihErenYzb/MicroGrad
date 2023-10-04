import math
class Value( ):
    def __init__(self, data,_creator=tuple()):
        self.data = data
        self.grad = 0.0
        self._creator = set(_creator)
        self._backward = lambda: None
    def __add__(self,other):
        out = Value(self.data+other.data,_creator= (self,other))
        def tmpbackward():
            self.grad += out.grad
            other.grad += out.grad


        out._backward = tmpbackward
        return out
    def tanh(self):
        x = self.data
        out = Value((math.exp(2*x)-1)/(math.exp(2*x)+1),_creator=(self,))
        def tmpbackward():
            self.grad += 1-out.data**2
        out._backward = tmpbackward
        return out
    def __mul__(self,other):
        out = Value(self.data*other.data,_creator=(self,other))
        def tmpbackward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = tmpbackward        
        return out
    def backward(self):
        self.grad = 1.0
        visited = set()
        tosort = []
        def toposort(x):
            if x not in visited:
                visited.add(x)
                for el in x._creator:
                    toposort(el)
                tosort.append(x)
        toposort(self)
        for i in reversed(tosort):
            i._backward()
        
            
            

    def __repr__(self):
        return f"{self.data}"



a = Value(3)
b = a+a
c = b.tanh()
c.backward()
print(a.grad)
print(b.grad)
print(c.grad)