class Value( ):
    def __init__(self, data,label=" "):
        self.data = data
        self.grad2 = 0
        self.creator = None
        self.op = None
    def __add__(self,other):
        tmp = Value(self.data+other.data)
        tmp.creator= (self,other)
        tmp.op = 1
        return tmp
    def __mul__(self,other):
        tmp = Value(self.data*other.data)
        tmp.creator= (self,other)
        tmp.op = 0
        return tmp

    def __repr__(self):
        return f"{self.data}"
    def grad(self,other=None,grad=1,highgrad=0):

        if other==None:
            self.grad2+=1*grad
            tmp=1*grad
        else:
            self.grad2+=other.data*grad
            tmp=other.data*grad

        if self.creator==None:
            
            return
        if self.op:
            self.creator[0].grad(grad=tmp)
            self.creator[1].grad(grad=tmp)

        else:
            self.creator[0].grad( self.creator[1],grad=tmp)
            self.creator[1].grad( self.creator[0],grad=tmp)
    

a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b    ; d.label = 'd'
e = a + b    ; e.label = 'e'
f = d * e    ; f.label = 'f'
f.grad()
print(a.grad2)
print(b.grad2)
print(d.grad2)
print(e.grad2)

print(f.grad2)





