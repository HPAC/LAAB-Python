# Report | LAAB-Python | 1xCPU-Core 

| Framework | TensorFlow/2.15.1-foss-2023a | 
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
|$A^TB$|`transpose(A)@B`| 0.522 | 0.056 | 0.016| :white_check_mark: |
|$"$|`linalg.matmul(t(A),B)` | 0.521 |  0.055 | 0.016 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **2.245**| **3.544** | | |
|**Ref (+)** |`sgemm`| **0.494**| - | | |



## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 0.529 | 0.0 | 0.0| :white_check_mark: |
|**Ref (-)** |`no cse`| **1.059**| **1.0** | | | 
|**Ref (+)**| `2*(transpose(A)@B)`| **0.529**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 1.054 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no cse`| **1.581**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.054**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.591 | 0.512 | 1.023 | :x: |
|**Ref (-)** |`no cse`| **1.579**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.053**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 1.585 | 279.12 | 1.514 | :x: |
|**Ref (-)** |`with cse`| **1.049**| **184.396** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.006**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.525 | 170.046 | 1.0 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.525**| **170.098** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.003**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.003 | 0.32 | 0.001 | :white_check_mark: |  
|**Ref (-)** |`eval. right to left`| **0.526**| **232.769** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.002**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.54 | 9.35 | 1.0 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.539**| **9.348** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.052**| - | | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.52 | 1.096 | 1.105 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.52 |  1.096 | 1.105  | :x: |
|**Ref (-)** |`sgemm`| **0.494**| **0.992** | | |
|**Ref (+)** |`trmm`| **0.248**| - | | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.536 | 1.111 | 1.176 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.536 | 1.11 | 1.175  | :x: |
|**Ref (-)** |`sgemm`| **0.494**| **0.945** | | |
|**Ref (+)** |`syrk`| **0.254**| - | | |


c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.52 | 116.487 |1.054 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.519 | 116.225 | 1.051  | :x: | 
|$"$|`linalg.tridiagonal_matmul(A,B)`| 0.023 | 4.128 | 0.037  | :white_check_mark: | 
|**Ref (-)** |`sgemm`| **0.494**| **110.538** | | |
|**Ref (+)** |`csr(A)@B`| **0.004**| - | | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 1.044 | 0.938 | 1.0 | :x: |
|**Ref (-)** |`no rewrite`| **1.044**| **0.938** | | |
|**Ref (+)**|`A@(B+C)`|**0.539**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.528 | 130.007 | 1.001 | :x: |
|**Ref (-)** |`no rewrite`| **0.528**| **129.921** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.004**| - | | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $ 1500 \times 1500 $.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$AB$|`A@B`| 0.521 | 0.922 | 1.003 | :x: |
|$"$|`linalg.matmul(A,B)` | 0.521 | 0.92 | 1.001 | :x: |
|**Ref (-)** |`no rewrite`| **0.521**| **0.919** | | |
|**Ref (+)**|`blocked matrix multiply`|**0.271**| - | | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.575 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.732**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.577**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.02 | 9.2 | 1.369 | :x: |
|**Ref (-)** |`no code motion`| **0.015**| **6.72** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.523 | 260.45 | 1.011 | :x: |
|**Ref (-)** |`no code motion`| **0.517**| **257.695** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.721 

### Score: 6 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />