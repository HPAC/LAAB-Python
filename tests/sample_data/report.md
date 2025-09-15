## LAAB-Python | CPU 

### Benchmark Information

| | |
|---|---|
| **Framework** | PyTorch/2.1.2-foss-2023a |
| **System** | HPC2N_x86_64 |
| **CPU** | AMD EPYC 7413 24-Core Processor | 


### Test 1: Comparison with GEMM

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | loss | result@0.05 | 
|----|------|------------|--|---|
|$A^TB$|`t(A)@B`| 0.509 | 0.031| :white_check_mark: |
|$"$|`linalg.matmul(t(A),B)` | 0.507 | 0.026 | :white_check_mark: |
|**Reference** |`sgemm`| **0.494**| | |

<hr style="border: none; height: 1px; background-color: #ccc;" />

### Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | loss | result@0.05 |
|-----|-----|----------|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | 0.523 | 0.0| :white_check_mark: | 
|**Reference**| `2*(t(A)@B)`| **0.526**| | |


b) **Repeated in multiplication**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$ and $E_3 = (A^TB)^TA^TB$. Evaluating these expressions from right to left involves three matrix multiplications. The reference implementation avoids the redundant computation of the common subexpression resulting in just two matrix multiplications.

|Expr|Call | time (s) | loss | result@0.05 |
|-----|-----|----------|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| 1.019 | 0.0 | :white_check_mark: |
|$E_3$|`t(t(A)@B)@t(A)@B`| 1.52 |  0.504 | :x: |
|**Reference**| `S=t(A)@B; t(S)@S`| **1.018**| | |

c) **Sub-optimal CSE**

TODO

<hr style="border: none; height: 1px; background-color: #ccc;" />

### Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{3000 \times 3000}$, $x \in \mathbb{R}^{3000}$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| loss | result@0.05 |
|----|----|---------|--|--|
|$H^THx$|`t(H)@H@x`| 0.509 | 94.417 | :x: |
|$"$|`linalg.multi_dot([t(H), H, x])`| 0.006 | 0.128 | :x: |
|**Reference**| `t(H)@(H@x)`| **0.005**| | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{3000 \times 3000}$, $y \in \mathbb{R}^{3000}$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| loss | result@0.05 |
|----|-----|---------|--|--|
|$y^TH^TH$|`t(y)@t(H)@H`| 0.006 | 0.0 | :white_check_mark: |
|$"$|`linalg.multi_dot([t(y), t(H), H])`| 0.006 | 0.0 | :white_check_mark: |
|**Reference**| `(t(y)@t(H))@H`| **0.006**| | |

c) **Mixed**:

Operands: $H \in \mathbb{R}^{3000 \times 3000}$ and $x,y \in \mathbb{R}^{3000}$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | loss | result@0.05 |
|----|----|-----------|--|--|
|$H^Tyx^TH$|`t(H)@y@t(x)@H`| 0.531 | 21.921 | :x: |
|$"$|`linalg.multi_dot([t(H), y, t(x), H])`| 0.024 | 0.0 | :white_check_mark: |
|**Reference**| `(t(H)@y)@(t(x)@H)`| **0.023**| | |

<hr style="border: none; height: 1px; background-color: #ccc;" />