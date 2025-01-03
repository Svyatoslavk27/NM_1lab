#Знайти найменший додатній корінь нелінійного рівняння х* - 5.7423 + 8.18х - 3.48 = 0 
# методом простої ітерації і модифікованим Нью-тона з точністю € = 10-4. 
# Знайти апріорну та апостеріорну оцінку кількості кроків. 
# Початковий проміжок та початкове наближення обрати однакове для обох методів (якщо це можливо),
#  порівняти результати роботи методів між собою.
import numpy as np

EPSILON = 1e-4  
MAX_ITERATIONS = 100 

def f(x):
    return x**4 - 5.74*x**3 + 8.18*x - 3.48

def f_derivative(x):
    return 4*x**3 - 17.22*x**2 + 8.18

def g(x):
    try:
        return (x**4 + 3.48) / (5.74*x**3 - 8.18)
    except ZeroDivisionError:
        return None  

def g_derivative(x):
    try:
        return abs(4 * x**3 / (5.74 * x**3 - 8.18) - (x**4 + 3.48) * (17.22 * x**2) / (5.74 * x**3 - 8.18)**2)
    except ZeroDivisionError:
        return None  

def simple_iteration(x0, epsilon=EPSILON, max_iterations=MAX_ITERATIONS):
    x = x0
    for i in range(max_iterations):
        x_new = g(x)
        if x_new is None:  
            print("Error: Division by zero during iteration")
            return None, i
        diff = abs(x_new - x)
        print(f"Iteration {i + 1}: x = {x_new}, difference = {diff}")
        if diff < epsilon:
            return x_new, i + 1
        x = x_new
    return None, max_iterations

def modified_newton(x0, epsilon=EPSILON, max_iterations=MAX_ITERATIONS):
    x = x0
    for i in range(max_iterations):
        f_val = f(x)
        f_deriv = f_derivative(x)
        if f_deriv == 0:
            print("Error: Derivative is zero during iteration")
            return None, i
        x_new = x - f_val / f_deriv
        diff = abs(x_new - x)
        print(f"Iteration {i + 1}: x = {x_new}, difference = {diff}")
        if diff < epsilon:
            return x_new, i + 1
        x = x_new
    return None, max_iterations

def apriori_estimate(q, x0, x1, epsilon):
    return np.ceil(np.log(abs(x1 - x0) / epsilon) / np.log(1 / q))

x0 = 2.2  

q = g_derivative(x0)
if q is not None and q >= 1:
    print("Метод простої ітерації не збіжний для заданого початкового наближення.")
else:
    print("Метод простої ітерації:")
    root_simple, iterations_simple = simple_iteration(x0)
    if root_simple is not None:
        print(f"Корінь: x = {root_simple}, кількість ітерацій: {iterations_simple}")
        if q is not None:
            apriori_steps = apriori_estimate(q, x0, root_simple, EPSILON)
            print(f"Апріорна оцінка кількості ітерацій: {apriori_steps}")

print("\nМетод модифікованого Ньютона:")
root_newton, iterations_newton = modified_newton(x0)
if root_newton is not None:
    print(f"Корінь: x = {root_newton}, кількість ітерацій: {iterations_newton}")
