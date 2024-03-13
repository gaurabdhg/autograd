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
        self.shape=data.shape
        self.grad:Optional['Tensor']=None
    
    def __repr__(self):
        return f'tensor({self.data},requiresGrad={self.requireGrad})'
    
    def sum(self):
        return tensorSum(self)
    
def tensorSum(t:Tensor)->Tensor:
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

  