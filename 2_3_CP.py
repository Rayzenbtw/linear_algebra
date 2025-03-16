import numpy as np
import pandas as pd
import sympy as sp

def gaussian_elimination(A):
    A = sp.Matrix(A)  # Используем sympy для точных вычислений
    rows, cols = A.shape
    
    for i in range(min(rows, cols)):
        max_row = max(range(i, rows), key=lambda r: abs(A[r, i]))
        if A[max_row, i] == 0:
            continue
        
        A.row_swap(i, max_row)
        A[i, :] = A[i, :] / A[i, i]
        
        for j in range(i + 1, rows):
            A[j, :] -= A[i, :] * A[j, i]
    
    return A

# Пример использования
A = [[2, -1, 1, 3], [1, 3, 2, 6], [3, 2, 4, 10], [1, 1, 1, 4]]
result = gaussian_elimination(A)
print(pd.DataFrame(np.array(result).astype(float)))
