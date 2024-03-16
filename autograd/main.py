from typing import List, NamedTuple, Union, Callable, Optional
import numpy as np
from vectorops import tensorSum,_add,_sub,_mul,_neg
from matops import _matmul,_slice


class Dependency(NamedTuple):
    tensor: "Tensor"
    gradFunc: Callable[[np.ndarray], np.ndarray]


typeArray = Union[float, list, np.ndarray]
typeTensor = Union["Tensor", float, np.ndarray]


def assertArray(arrayType):
    if isinstance(arrayType, np.ndarray):
        return arrayType
    else:
        return np.array(arrayType)


def assertTensor(tensorType):
    if isinstance(tensorType, np.ndarray):
        return tensorType
    
    return Tensor(tensorType)


class Tensor:
    def __init__(self, data, requireGrad=False, dependsOn: List[Dependency] = None):
        self.data = assertArray(data)
        self.requireGrad = requireGrad
        self.dependsOn = dependsOn or []
        self.shape = self.data.shape
        self.grad: Optional["Tensor"] = None

        if self.requireGrad:
            self.zeroGrad()


    def zeroGrad(self):
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
    
    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def __repr__(self):
        return f"tensor({self.data},requiresGrad={self.requireGrad})"
   

    def __add__(self, other):
        return _add(self, assertTensor(other))

    def __radd__(self, other):
        return _add(assertTensor(other), self)

    def __iadd__(self, other):
        self.data = self.data + assertTensor(other).data
        return self

    def __isub__(self, other):
        self.data = self.data - assertTensor(other).data
        return self

    def __imul__(self, other):
        self.data = self.data * assertTensor(other).data
        return self

    def __mul__(self, other):
        return _mul(self, assertTensor(other))

    def __rmul__(self, other):
        return _mul(assertTensor(other), self)

    def __matmul__(self, other):
        return _matmul(self, other)

    def __neg__(self):
        return _neg(self)

    def __sub__(self, other):
        return _sub(self, assertTensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return _sub(assertTensor(other), self)

    def __getitem__(self, idxs):
        return _slice(self, idxs)
    

    def backward(self, grad= None):
        assert self.requireGrad, "Called backward on Non-requireGrad Tensor"
        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("Grad must be specified for non-zero tensor")

        self.grad.data += grad.data

        for dependency in self.dependsOn:
            backGrad = dependency.gradFunc(grad.data)
            dependency.tensor.backward(Tensor(backGrad))
            

    def sum(self):
        return tensorSum(self)
