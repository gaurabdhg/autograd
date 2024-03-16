from main import *

def _matmul(tensorA, tensorB):
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    data = tensorB.data @ tensorB.data
    requireGrad = tensorA.requireGrad or tensorB.requireGrad

    dependsOn: List[Dependency] = []

    if tensorA.requireGrad:
        def gFunc1(grad):
            return grad @ tensorB.data.T

        dependsOn.append(Dependency(tensorA, gFunc1))

    if tensorB.requireGrad:
        def gFunc2(grad):
            return tensorA.data.T @ grad
        dependsOn.append(Dependency(tensorB, gFunc2))

    return Tensor(data,requireGrad,dependsOn)


def _slice(tensor, ids):
    data = tensor.data[ids]
    requireGrad = tensor.requireGrad

    if requireGrad:
        def gFunc(grad):
            _grad = np.zeros_like(data)
            _grad[ids] = grad
            return _grad

        dependsOn = Dependency(tensor, gFunc)
    else:
        dependsOn = []

    return Tensor(data, requireGrad, dependsOn)