from main import *


def tensorSum(tensor):
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = tensor.data.sum()
    requireGrad = tensor.requireGrad
    if requireGrad:

        def gradFunc(grad: np.ndarray):
            return grad * np.ones_like(tensor.data)

        dependsOn = [Dependency(tensor, gradFunc)]
    else:
        dependsOn = []

    return Tensor(data, requireGrad, dependsOn)


def _add(tensorA, tensorB):
    data = tensorA.data + tensorB.data
    requireGrad = tensorA.requireGrad or tensorB.requireGrad
    dependsOn: List[Dependency] = []

    if tensorA.requireGrad:

        def gFunc1(grad):
            ndSum = grad.ndim - tensorA.data.ndim
            for _ in range(ndSum):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(tensorA.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

    dependsOn.append(Dependency(tensorA, gFunc1))

    if tensorB.requireGrad:

        def gFunc2(grad):
            ndSum = grad.ndim - tensorB.data.ndim
            for _ in range(ndSum):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(tensorB.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        dependsOn.append(Dependency(tensorB, gFunc2))

    return Tensor(data, requireGrad, dependsOn)

def _mul(tensorA, tensorB):
    data = tensorA.data + tensorB.data
    requireGrad = tensorA.requireGrad or tensorB.requireGrad
    dependsOn: List[Dependency] = []

    if tensorA.requireGrad:

        def gFunc1(grad):
            grad=grad*tensorB.data
            ndSum = grad.ndim - tensorA.data.ndim
            for _ in range(ndSum):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(tensorA.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

    dependsOn.append(Dependency(tensorA, gFunc1))

    if tensorB.requireGrad:

        def gFunc2(grad):
            
            grad=grad*tensorA.data
            ndSum = grad.ndim - tensorB.data.ndim
            for _ in range(ndSum):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(tensorB.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        dependsOn.append(Dependency(tensorB, gFunc2))

    return Tensor(data, requireGrad, dependsOn)

def _neg(tensor):
    data = -tensor.data
    requireGrad = tensor.requireGrad
    if requireGrad:
        dependsOn = [Dependency(tensor, lambda x: -x)]
    else:
        dependsOn = []

    return Tensor(data, requireGrad, dependsOn)

def _sub(tensorA, tensorB):
    return _add(tensorA + _neg(tensorB))

