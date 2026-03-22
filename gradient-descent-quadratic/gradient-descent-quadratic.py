
def differentiate(a, b, c, x0):
    return 2 * a * x0 + b

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    for i in range(steps):
        d = differentiate(a, b, c, x0)
        x0 = x0 - lr * d

    return x0