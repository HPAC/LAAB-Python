# Report | LAAB-Python | 1xCPU-Core 

| Framework | Jax/0.4.25-gfbf-2023a | 
|---|---|
| **System** | HPC2N_x86_64 |
| **CPU** | AMD EPYC 7413 24-Core Processor | 
| **LAAB_N** | 3000 |
| **LAAB_REP** | 10 |
| **OMP_NUM_THREADS** | 1 |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Comparison with GEMM

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | loss | result@0.10 | 
|----|------|------------|--|---|
|$A^TB$|`transpose(A)@B`| 0.507 | 0.035| :white_check_mark: |
|$"$|`jax.numpy.matmul(t(A),B)` |  |  |  |
|**Reference** |`sgemm`| **0.495**| | |


## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 1.015 | 0.0| :white_check_mark: | 
|**Reference**| `2*(transpose(A)@B)`| **1.016**| | |


b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 1.524 | 0.513 | :x: |
|**Reference**| `S=transpose(A)@B; transpose(S)@S`| **1.015**| | |

c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.506 |  0.513 | :x: |
|**Reference**| `S=transpose(A)@B; transpose(S)@S`| **1.015**| | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 1.506 |  36.396 | :x: |
|**Reference**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.04**| | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| loss | result@0.10 |
|----|----|---------|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.509 | 24.818 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), H, x])`| 0.02 | 0.04 | :white_check_mark: |  
|**Reference**| `transpose(H)@(H@x)`| **0.019**| | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| loss | result@0.10 |
|----|-----|---------|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.011 | 0.039 | :white_check_mark: |  
|$"$|`linalg.multi_dot([transpose(y), transpose(H), H])`| 0.011 | 0.024 | :white_check_mark: | 
|**Reference**| `(transpose(y)@transpose(H))@H`| **0.011**| | |

c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | loss | result@0.10 |
|----|----|-----------|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.543 | 14.231 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), y, transpose(x), H])`| 0.037 | 0.062 | :white_check_mark: | 
|**Reference**| `(transpose(H)@y)@(transpose(x)@H)`| **0.036**| | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | loss | result@0.10|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.507 | 1.078 | :x: |
|$"$|`jax.numpy.matmul(A,B)`|  |   |  |
|**Reference** |`trmm`| **0.248**| | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | loss | result@0.10|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.507 | 0.997 | :x: |
|$"$|`jax.numpy.matmul(A,B)`|  |   |  |
|**Reference** |`syrk`| **0.255**| | |

c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | loss | result@0.10|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.51 | 115.495 | :x: |
|$"$|`jax.numpy.matmul(A,B)`|  |   |  | 
|**Reference** |`csr(A)@B`| **0.004**| | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
|$E_1$|`A@B+ A@C`| 1.03 | 0.934| :x: |
|**Reference**|`A@(B+C)`|**0.532**| | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.516 | 22.734| :x: |
|**Reference**|`A@x - transpose(H)@(H@x)`|**0.022**| | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $ 1500 \times 1500 $.

|Expr|Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
|$AB$|`A@B`| 0.507 | 0.775 | :x: |
|$"$|`jax.numpy.matmul(A,B)` |  |  |  |
|**Reference**|`blocked matrix multiply`|**0.281**| | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
||`for i in range(3): A@B ...`| 0.526 |  0.0 | :white_check_mark: |
|**Reference**|`A@B; for i in range(3): ...`|**0.527**| | | 

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
||`(A+B)[2,2]`| 0.0 | 0.0 | :white_check_mark: |
|**Reference**|`A[2,2] + B[2,2]`|**0.002**| | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
||`(A@B)[2,2]`| 0.509 | 259.1 | :x: |
|**Reference**|`dot(A[2,:],B[:,2])`|**0.002**| | |


## OVERALL RESULT

### Mean loss: 25.740 

### Score: 7 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />