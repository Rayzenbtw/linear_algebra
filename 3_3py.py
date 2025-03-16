import numpy as np
import pandas as pd
import sympy
import matplotlib.pyplot as plt
from scipy import linalg

def cramer_rule(A, b):
    """
    Розв'язання системи лінійних рівнянь методом Крамера.
    
    Args:
        A: Матриця коефіцієнтів системи
        b: Вектор правої частини
        
    Returns:
        list: Розв'язок системи або повідомлення про помилку
    """
    n = len(A)
    # Перетворюємо вхідні дані на матриці NumPy для швидшого обчислення
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float).reshape(n, 1)
    
    # Обчислюємо головний визначник
    det_A = np.linalg.det(A_np)
    
    # Перевіряємо, чи можемо застосувати метод Крамера
    if abs(det_A) < 1e-10:  # Використовуємо малу величину для уникнення проблем з точністю
        return "Система не має єдиного розв'язку (головний визначник ≈ 0)"
    
    # Обчислюємо розв'язок за формулами Крамера
    solutions = []
    for i in range(n):
        # Створюємо копію матриці A
        A_i = A_np.copy()
        # Замінюємо i-й стовпець на вектор правої частини
        A_i[:, i] = b_np.flatten()
        # Обчислюємо визначник i-ї матриці
        det_A_i = np.linalg.det(A_i)
        # Обчислюємо i-у змінну
        x_i = det_A_i / det_A
        solutions.append(x_i)
    
    return solutions

def run_benchmark(size):
    """
    Проводить тестування методів розв'язку на матриці заданого розміру.
    
    Args:
        size: Розмір матриці (n)
        
    Returns:
        dict: Результати тестування
    """
    print(f"\n====== Тестування матриці {size}x{size} ======")
    
    # Створюємо випадкову матрицю з невеликими цілими числами для кращої стабільності
    A = np.random.randint(-5, 5, (size, size))
    b = np.random.randint(-10, 10, size)
    
    print("Матриця A:")
    print(A)
    print("\nВектор b:")
    print(b)
    
    # Розв'язання за допомогою методу Крамера
    print("\nРозв'язання методом Крамера...")
    cramer_solution = cramer_rule(A, b)
    
    if isinstance(cramer_solution, str):
        print(f"Результат: {cramer_solution}")
        return {'size': size, 'result': 'Система не має єдиного розвязку'}
    
    print("Розв'язок x:")
    for i, val in enumerate(cramer_solution):
        print(f"x{i+1} = {val}")
    
    # Розв'язання за допомогою NumPy
    try:
        print("\nРозв'язання за допомогою numpy.linalg.solve...")
        np_solution = np.linalg.solve(A, b)
        print("Розв'язок x (NumPy):")
        for i, val in enumerate(np_solution):
            print(f"x{i+1} = {val}")
        
        # Порівняння точності
        max_diff = np.max(np.abs(np.array(cramer_solution) - np_solution))
        print(f"\nМаксимальна різниця між розв'язками: {max_diff}")
        
        # Перевірка розв'язку (A*x = b)
        residual_cramer = np.max(np.abs(A.dot(cramer_solution) - b))
        residual_numpy = np.max(np.abs(A.dot(np_solution) - b))
        print(f"Залишок для методу Крамера (|Ax - b|): {residual_cramer}")
        print(f"Залишок для методу NumPy (|Ax - b|): {residual_numpy}")
        
    except np.linalg.LinAlgError:
        print("NumPy: Система не має єдиного розв'язку")
    
    # Теоретична оцінка кількості операцій
    cramer_ops = (size + 1) * (size**3)
    numpy_ops = (2/3) * (size**3)
    ratio = cramer_ops / numpy_ops
    
    print(f"\nТеоретична кількість операцій:")
    print(f"Метод Крамера: {cramer_ops}")
    print(f"Метод NumPy: {numpy_ops}")
    print(f"Метод Крамера теоретично повільніший у {ratio:.2f} разів")
    
    return {
        'size': size,
        'cramer_solution': cramer_solution,
        'numpy_solution': np_solution if 'np_solution' in locals() else None,
        'max_difference': max_diff if 'max_diff' in locals() else None,
        'residual_cramer': residual_cramer if 'residual_cramer' in locals() else None,
        'residual_numpy': residual_numpy if 'residual_numpy' in locals() else None,
        'theoretical_ops_cramer': cramer_ops,
        'theoretical_ops_numpy': numpy_ops,
        'theoretical_speedup': ratio
    }

def test_specific_cases():
    """
    Тестує спеціальні випадки систем лінійних рівнянь.
    """
    print("\n====== Тестування спеціальних випадків ======")
    
    # Випадок 1: Система з нульовим визначником
    print("\nВипадок 1: Система з нульовим визначником")
    A_singular = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    b_singular = np.array([1, 2, 3])
    
    print("Матриця A:")
    print(A_singular)
    print("Визначник A:", np.linalg.det(A_singular))
    
    cramer_result = cramer_rule(A_singular, b_singular)
    print(f"Результат методу Крамера: {cramer_result}")
    
    try:
        np_result = np.linalg.solve(A_singular, b_singular)
        print(f"Результат NumPy: {np_result}")
    except np.linalg.LinAlgError as e:
        print(f"Помилка NumPy: {e}")
    
    # Випадок 2: Система з точним розв'язком
    print("\nВипадок 2: Система з точним розв'язком (цілими числами)")
    A_exact = np.array([
        [2, 1, -1],
        [3, 4, 2],
        [1, -3, 1]
    ])
    b_exact = np.array([3, 1, 2])
    
    print("Матриця A:")
    print(A_exact)
    print("Вектор b:", b_exact)
    
    cramer_result = cramer_rule(A_exact, b_exact)
    print(f"Результат методу Крамера: {cramer_result}")
    
    np_result = np.linalg.solve(A_exact, b_exact)
    print(f"Результат NumPy: {np_result}")

# Тестування на матрицях розмірами 10×10 та 12×12
benchmark_results = []

sizes_to_test = [10, 12]
for size in sizes_to_test:
    result = run_benchmark(size)
    benchmark_results.append(result)

# Порівняння теоретичної ефективності для різних розмірів
sizes_theory = list(range(2, 16, 2))
theory_results = []

for size in sizes_theory:
    cramer_ops = (size + 1) * (size**3)
    numpy_ops = (2/3) * (size**3)
    ratio = cramer_ops / numpy_ops
    
    theory_results.append({
        'size': size,
        'theoretical_ops_cramer': cramer_ops,
        'theoretical_ops_numpy': numpy_ops,
        'theoretical_speedup': ratio
    })

# Таблиця для теоретичного порівняння
theory_df = pd.DataFrame(theory_results)
print("\n====== Теоретичне порівняння ефективності ======")
print(theory_df)

# Побудова графіків
plt.figure(figsize=(10, 6))
plt.plot(sizes_theory, [r['theoretical_ops_cramer'] for r in theory_results], 'o-', label='Метод Крамера')
plt.plot(sizes_theory, [r['theoretical_ops_numpy'] for r in theory_results], 's-', label='Метод NumPy')
plt.xlabel('Розмір матриці (n)')
plt.ylabel('Теоретична кількість операцій')
plt.title('Порівняння обчислювальної складності')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.figure(figsize=(10, 6))
plt.plot(sizes_theory, [r['theoretical_speedup'] for r in theory_results], 'o-')
plt.xlabel('Розмір матриці (n)')
plt.ylabel('Відносне прискорення')
plt.title('У скільки разів метод NumPy теоретично швидший за метод Крамера')
plt.grid(True)

# Додатково протестуємо систему з особливими випадками
test_specific_cases()

plt.tight_layout()
plt.show()