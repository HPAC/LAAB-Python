# Report | LAAB-Python | 1xCPU-Core 

| Framework | Jax/0.4.25-gfbf-2023a | 
|---|---|
| **System** | HPC2N_x86_64 |
| **CPU** | AMD EPYC 7413 24-Core Processor | 
| **LAAB_N** | 3000 |
| **LAAB_REP** | 10 |
| **OMP_NUM_THREADS** | 1 |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Matrix multiplications

The execution times of matrix multiplications invoked through the high-level APIs of the frameworks are compared against those of an optimised reference implementation.

a) **GEMM**:

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | slowdown | loss | result@0.05 | 
|----|------|------------|--|---|--|
|$A^TB$|`transpose(A)@B`| 0.507 | 0.027 | 0.008| :white_check_mark: |
|$"$|`jax.numpy.matmul(transpose(A),B)` | 0.508 |  0.028 | 0.008 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **2.25**| **3.555** | | |
|**Ref (+)** |`sgemm`| **0.494**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.508 | 1.065 | 1.056 | :x: |
|$"$|`jax.numpy.matmul(L,B)`| 0.508 |  1.064 | 1.055  | :x: |
|**Ref (-)** |`sgemm`| **0.494**| **1.008** | | |
|**Ref (+)** |`trmm`| **0.246**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@transpose(A)`| 0.507 | 1.013 | 1.055 | :x: |
|$"$|`jax.numpy.matmul(A,transpose(A))`| 0.507 | 1.013 | 1.055  | :x: |
|**Ref (-)** |`sgemm`| **0.494**| **0.96** | | |
|**Ref (+)** |`syrk`| **0.252**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.509 | 116.973 |1.03 | :x: |
|$"$|`jax.numpy.matmul(T,B)`| 0.509 | 117.068 | 1.031  | :x: | 
|**Ref (-)** |`sgemm`| **0.494**| **113.564** | | |
|**Ref (+)** |`csr(T)@B`| **0.004**| - | | |

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
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 1.521 | 0.501 | 1.003 | :x: |
|**Ref (-)** |`no cse`| **1.52**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.013**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.507 | 0.486 | 0.972 | :x: |
|**Ref (-)** |`no cse`| **1.521**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.014**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 1.506 | 41.018 | 1.496 | :x: |
|**Ref (-)** |`with cse`| **1.018**| **27.413** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.036**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.508 | 25.233 | 0.999 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), H, x])`| 0.02 | 0.052 | 0.002 | :white_check_mark: |  
|**Ref (-)** |`eval. left to right`| **0.508**| **25.264** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.019**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.011 | 0.026 | 0.001 | :white_check_mark: |  
|$"$|`linalg.multi_dot([transpose(y), transpose(H), H])`| 0.011 | 0.005 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. right to left`| **0.508**| **46.899** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.011**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.543 | 14.805 | 1.002 | :x: | 
|$"$|`linalg.multi_dot([transpose(H), y, transpose(x), H])`| 0.035 | 0.027 | 0.002 | :white_check_mark: | 
|**Ref (-)** |`eval. left to right`| **0.542**| **14.777** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.034**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 1.019 | 0.928 | 0.999 | :x: |
|**Ref (-)** |`no rewrite`| **1.02**| **0.929** | | |
|**Ref (+)**|`A@(B+C)`|**0.529**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.518 | 23.197 | 0.999 | :x: |
|**Ref (-)** |`no rewrite`| **0.519**| **23.23** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.021**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_3$|`transpose(B)@A@transpose(A)@B`| 1.506 | 0.484 | 0.999 | :x: |
|**Ref (-)** |`no rewrite`| **1.507**| **0.485** | | |
|**Ref (+)**|`S = transpose(A)@B; transpose(S)@S`|**1.015**| - | | |

d) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks $A_1$ and $A_2$ along the diagnonal, each of size $ 1500 \times 1500 $, and the remaining elements are zero. The result of the matrix multiplication $AB$ can be rewritten as $[(A_1B_1), (A_2B_2)]$, where $B_1, B_2$ are of sizes $1500 \times 3000$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$AB$|`A@B`| 0.509 | 0.792 | 1.002 | :x: |
|$"$|`jax.numpy.matmul(A,B)` | 0.509 | 0.793 | 1.004 | :x: |
|**Ref (-)** |`no rewrite`| **0.509**| **0.79** | | |
|**Ref (+)**|`blocked matrix multiply`|**0.284**| - | | |


## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.528 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.585**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.528**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.002 | 0.088 | 0.011 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **0.018**| **8.045** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.511 | 254.615 | 1.01 | :x: |
|**Ref (-)** |`no code motion`| **0.506**| **252.005** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.702 

### Score: 6 / 18

<hr style="border: none; height: 1px; background-color: #ccc;" />