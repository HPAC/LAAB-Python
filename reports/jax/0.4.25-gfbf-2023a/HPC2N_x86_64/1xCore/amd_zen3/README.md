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


||Call  |  time (s)  | slowdown | loss | result@0.05 | 
|----|------|------------|--|---|--|
|$A^TB$|`transpose(A)@B`| 0.508 | 0.027 | 0.008| :white_check_mark: |
|$"$|`jax.numpy.matmul(t(A),B)` | 0.507 |  0.027 | 0.008 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **2.234**| **3.522** | | |
|**Ref (+)** |`sgemm`| **0.494**| - | | |



## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 1.018 | 0.992 | 0.992| :x: |
|**Ref (-)** |`no cse`| **1.022**| **1.0** | | | 
|**Ref (+)**| `2*(transpose(A)@B)`| **0.511**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 1.523 | 0.501 | 1.002 | :x: |
|**Ref (-)** |`no cse`| **1.522**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.015**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.512 | 0.487 | 0.974 | :x: |
|**Ref (-)** |`no cse`| **1.525**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.017**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 1.514 | 38.282 | 1.51 | :x: |
|**Ref (-)** |`with cse`| **1.015**| **25.346** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.039**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.507 | 25.19 | 1.0 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), H, x])`| 0.02 | 0.054 | 0.002 | :white_check_mark: |  
|**Ref (-)** |`eval. left to right`| **0.507**| **25.187** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.019**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.011 | 0.045 | 0.001 | :white_check_mark: |  
|$"$|`linalg.multi_dot([transpose(y), transpose(H), H])`| 0.011 | 0.001 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. right to left`| **0.507**| **46.59** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.011**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.543 | 14.166 | 1.0 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), y, transpose(x), H])`| 0.037 | 0.023 | 0.002 | :white_check_mark: | 
|**Ref (-)** |`eval. left to right`| **0.543**| **14.172** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.036**| - | | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.507 | 1.052 | 1.052 | :x: |
|$"$|`jax.numpy.matmul(A,B)`| 0.508 |  1.055 | 1.055  | :x: |
|**Ref (-)** |`sgemm`| **0.494**| **1.0** | | |
|**Ref (+)** |`trmm`| **0.247**| - | | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.506 | 1.008 | 1.05 | :x: |
|$"$|`jax.numpy.matmul(A,B)`| 0.506 | 1.009 | 1.05  | :x: |
|**Ref (-)** |`sgemm`| **0.494**| **0.96** | | |
|**Ref (+)** |`syrk`| **0.252**| - | | |


c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.508 | 114.388 |1.028 | :x: |
|$"$|`jax.numpy.matmul(A,B)`| 0.509 | 114.66 | 1.03  | :x: | 
|**Ref (-)** |`sgemm`| **0.494**| **111.298** | | |
|**Ref (+)** |`csr(A)@B`| **0.004**| - | | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 1.018 | 0.931 | 0.998 | :x: |
|**Ref (-)** |`no rewrite`| **1.019**| **0.934** | | |
|**Ref (+)**|`A@(B+C)`|**0.527**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.516 | 23.413 | 1.0 | :x: |
|**Ref (-)** |`no rewrite`| **0.516**| **23.411** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.021**| - | | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $ 1500 \times 1500 $.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$AB$|`A@B`| 0.507 | 0.801 | 1.001 | :x: |
|$"$|`jax.numpy.matmul(A,B)` | 0.507 | 0.802 | 1.002 | :x: |
|**Ref (-)** |`no rewrite`| **0.507**| **0.8** | | |
|**Ref (+)**|`blocked matrix multiply`|**0.281**| - | | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.526 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.578**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.526**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.002 | 0.205 | 0.029 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.016**| **7.19** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.51 | 253.765 | 1.009 | :x: |
|**Ref (-)** |`no code motion`| **0.505**| **251.525** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.686 

### Score: 6 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />