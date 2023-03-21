import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


# Derivatives Overview
# def f(x):
#     return 3*x**2 - 4*x + 5
# print(f(3.0))

# xs = np.arange(-5, 5, 0.25)
# print(xs)
# ys = f(xs)
# print(ys)
# plt.plot(xs, ys)
# plt.show()

# h = 0.001
# x = 3.0
# f(x + h)
# print(f(x + h))
# f(x + h) - f(x)
# print(f(x + h) - f(x))
# (f(x + h) - f(x)) / h # Gives us the slope
# print((f(x + h) - f(x)) / h)

# Next Example
# h = 0.0001

# inputs
# a = 2.0
# b = -3.0
# c = 10

# d1 = a*b + c
# a += h
# d2 = a*b + c
# print(d)
# print('d1', d1)
# print('d2', d2)
# print('slope', (d2-d1)/h)

# First Simple Value Object
class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b;
e.label = 'e'
d = e + c;
d.label = 'd'
f = Value(-2.0, label='f')
L = d * f;
L.label = 'L'
L
# print(L)


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

draw_dot(L)

a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

e = a * b
d = e + c
L = d * f

print(L.data)

# Gating Function Protects Global Scope
def lol():
    h = 0.001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b;
    e.label = 'e'
    d = e + c;
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f;
    L.label = 'L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    b.data += h
    c = Value(10.0, label='c')
    e = a * b;
    e.label = 'e'
    d = e + c;
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f;
    L.label = 'L'
    L2 = L.data

    print((L2 - L1) / h)


print(lol())
