from typing import List,NamedTuple,Union,Callable,Optional
import numpy as np


class Dependency(NamedTuple):
    tensor: 'Tensor'
    gradFunc: Callable[[np.ndarray], np.ndarray]

typeArray=Union[float,list,np.ndarray]
typeTensor=Union['Tensor',float,np.ndarray]

def assertArray(arrayType:typeArray)->np.ndarray:
    if isinstance(arrayType,np.ndarray):
        return arrayType
    else:
        return np.array(arrayType)


def assertTensor(tensorType:typeTensor)->'Tensor':
    if isinstance(tensorType,np.ndarray):
        return tensorType
    else:
        return Tensor(tensorType)

class Tensor:
    def __init__(self,data,requireGrad=False,dependsOn:List[Dependency]=None):
        self.data=assertArray(data)
        self.requireGrad=requireGrad
        self.dependsOn-=dependsOn or []
        self.shape=self.data.shape
        self.grad:Optional['Tensor']=None
        
        if self.requireGrad:
            self.zeroGrad()
    
    
    def __repr__(self):
        return f'tensor({self.data},requiresGrad={self.requireGrad})'
    
    def backward(self,grad:'Tensor'=None):
        assert self.requireGrad,"Called backward on Non-requireGrad Tensor"
        if grad is None:
            if self.shape==():
                grad=Tensor(1)
            else:
                raise RuntimeError("Grad must be specified for non-zero tensor")
            
        self.grad.data+=grad.data
        
        for dependency in self.dependsOn:
            backGrad=dependency.gradFunc(grad.data)
            dependency.tensor.backward(Tensor(backGrad))
    
    def sum(self):
        return tensorSum(self)
    
    def zeroGrad(self):
        self.grad=Tensor(np.zeros_like(self.data,dtype=np.float64))
        
def tensorSum(t):
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    sum=t.data.sum()
    requireGrad=t.requireGrad
    if requireGrad:
        def gradFunc(grad:np.ndarray):
            return grad*np.ones_like(t.data)
        dependsOn=[Dependency(t,gradFunc)]
    else:
        dependsOn=[]
    
    return Tensor(sum,requireGrad,dependsOn)

def _add(t1:Tensor,t2:Tensor):
    data=t1.data+t2.data
    requireGrad=t1.requireGrad or t2.requireGrad
    dependsOn:List[Dependency]=[]
    
    if t1.requireGrad:
        