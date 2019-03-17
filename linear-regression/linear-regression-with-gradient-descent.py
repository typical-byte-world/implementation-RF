import numpy as np

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]



def gradien_descent_runner(data, start_b, start_m, learning_rate, num_iterations):
    b = start_b
    m = start_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, data, learning_rate)

    return [b, m]
def run():
    data = np.genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    #y = mx + b
    init_m = 0
    init_b = 0
    num_iterations = 1000
    print(f"Intial value of b is {init_b}.\nInitial value of m is {init_m}\nIterations {num_iterations}")

    [m, b] = gradien_descent_runner(data, init_b, init_m, learning_rate, num_iterations)
    print(f"\nAfter {num_iterations} iterations end value of b is {b} and value of m is {m}")


if __name__ == '__main__':
    run()

