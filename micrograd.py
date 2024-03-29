import math
class Value( ):
    def __init__(self, data,_creator=tuple(),op = " "):
        self._op = op
        self.data = data
        self.grad = 0.0
        self._creator = _creator # why used set before?????
    # REMINDER: when calculating gradient, 
    # gradient of other is not calculated in pow function
    # even if it is an instance of Value
    def __pow__(self,other):
        other = other if not isinstance(other,Value) else other.data
        out  = Value(self.data**other,_creator=(self,other),op="**")
        return out
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,_creator= (self,other),op="+")
        return out
    def __neg__(self):
        return self * -1
    def __radd__(self,other):
        return self+other
    def __rsub__(self,other):
        return other + (- self)
    def __sub__(self,other):
        return self + (- other) 
    def tanh(self):
        x = self.data
        out = Value((math.exp(2*x)-1)/(math.exp(2*x)+1),_creator=(self,),op="tanh")
        return out
    def exp(self):
        out = Value(math.exp(self.data),_creator=(self,),op="exp")
        return out
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,_creator=(self,other),op="*")   
        return out
    def _backward(self):
        if self._op == "+":
            self._creator[0].grad += self.grad
            self._creator[1].grad += self.grad
        elif self._op == "**":
            self._creator[0].grad += self.grad*self._creator[1]*(self._creator[0].data**(self._creator[1]-1))
        elif self._op == "*":
            self._creator[0].grad += self.grad*self._creator[1].data
            self._creator[1].grad += self.grad*self._creator[0].data
        elif self._op == "exp":
            self._creator[0].grad += self.grad*self.data
        elif self._op == "tanh":
            self._creator[0].grad += (1-self.data**2)*self.grad
        else:
            # this part is executed when there are no _creators
            pass

    def backward(self):
        self.grad = 1.0
        visited = set()
        tosort = []
        def toposort(x):
            if x not in visited and isinstance(x,Value):
                visited.add(x)
                for el in x._creator:
                    toposort(el)
                tosort.append(x)
        toposort(self)
        for i in reversed(tosort):
            i._backward()
    def __repr__(self):
        return f" Value,data:{self.data}"