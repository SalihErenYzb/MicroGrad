from micrograd import Value
from random import uniform

class Neuron:
    def __init__(self,inpsize):
        self.w = [Value(uniform(-1,1)) for _ in range(inpsize)]
        self.b =Value( uniform(-1,1))
    def __call__(self,inp):
        return sum((w1*z1 for w1,z1 in zip(self.w,inp)),self.b).tanh()
    def _parameters(self):
        return self.w +[self.b]
class Layer:
    def __init__(self,inpsize,outsize):
        self.neurons = [Neuron(inpsize) for _ in range(outsize)]
    def __call__(self,inp):
        return [self.neurons[i](inp) for i in range(len(self.neurons))]
    def _parameters(self):
        return [param for neuron in self.neurons for param in neuron._parameters()]
class MLP:
    def __init__(self,inpsize,layersizes):
        sizes = [inpsize]+layersizes
        self.layers = [Layer(sizes[i],sizes[i+1]) for i in range(len(layersizes))]
    def __call__(self,inp):
        for i in self.layers:
            inp =i(inp)
        return  inp[0] if len(inp)==1 else inp
    def parameters(self):
        return [param for layer in self.layers for param in layer._parameters()]

mlp = MLP(4,[4,4,1])
xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1,.0,1.0],
    [1.0,1.0,-1.0]
]
ys = [1.0,-1.0,-1.0,1.0]
    
for id in range(1000):
    #forward
    outs = [mlp(x) for x in xs]
    loss = sum([(out-g)**2 for out,g in zip(outs,ys)])
    print(loss)
    #zerograd
    for r in mlp.parameters():
        r.grad = 0.0    

    #backward propagation
    loss.backward()
    #update
    lr = 0.2
    for r in mlp.parameters():
        r.data -= lr*r.grad
    
