import math
from nn import Value


def test_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b

    c.backward()

    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b

    c.backward()

    assert c.data == 6.0
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_power():
    a = Value(3.0)
    b = a**2

    b.backward()

    assert b.data == 9.0
    assert a.grad == 6.0


def test_tanh():
    a = Value(0.0)
    b = a.tanh()

    b.backward()

    assert math.isclose(b.data, 0.0, abs_tol=1e-7)
    assert math.isclose(a.grad, 1.0, abs_tol=1e-7)


def test_combined_expression():
    x = Value(2.0)
    y = Value(-3.0)

    z = x**2 + x * y + y**2
    z.backward()

    assert z.data == 7.0
    assert x.grad == 1.0
    assert y.grad == -4.0