# Report | LAAB-Python | 1xCore 

| Framework | TensorFlow/2.15.1-foss-2023a | 
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
|$A^TB$|`transpose(A)@B`| 0.521 | 0.033 | 0.01| :white_check_mark: |
|$"$|`linalg.matmul(transpose(A),B)` | 0.521 |  0.034 | 0.01 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **2.136**| **3.239** | | |
|**Ref (+)** |`sgemm`| **0.504**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.521 | 1.068 | 1.068 | :x: |
|$"$|`linalg.matmul(L,B)`| 0.521 |  1.069 | 1.069  | :x: |
|**Ref (-)** |`sgemm`| **0.504**| **1.0** | | |
|**Ref (+)** |`trmm`| **0.252**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@transpose(A)`| 0.528 | 1.094 | 1.094 | :x: |
|$"$|`linalg.matmul(A,transpose(A))`| 0.528 | 1.094 | 1.094  | :x: |
|**Ref (-)** |`sgemm`| **0.504**| **1.0** | | |
|**Ref (+)** |`syrk`| **0.252**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.521 | 118.057 |1.035 | :x: |
|$"$|`linalg.matmul(T,B)`| 0.521 | 117.881 | 1.033  | :x: | 
|$"$|`linalg.tridiagonal_matmul(T,B)`| 0.023 | 4.313 | 0.038  | :white_check_mark: | 
|**Ref (-)** |`sgemm`| **0.504**| **114.068** | | |
|**Ref (+)** |`csr(T)@B`| **0.004**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`transpose(A)@B + transpose(A)@B` | 0.53 | 0.0 | 0.0| :white_check_mark: |
|**Ref (-)** |`no cse`| **1.061**| **1.0** | | | 
|**Ref (+)**| `2*(transpose(A)@B)`| **0.531**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 1.053 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no cse`| **1.579**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.052**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.592 | 0.513 | 1.026 | :x: |
|**Ref (-)** |`no cse`| **1.578**| **0.5** | | |
|**Ref (+)**| `S=transpose(A)@B; transpose(S)@S`| **1.052**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`transpose(A)@B@transpose(A)@B@y`| 1.586 | 289.548 | 1.515 | :x: |
|**Ref (-)** |`with cse`| **1.049**| **191.163** | | |
|**Ref (+)**| `transpose(A)@(B@(transpose(A)@(B@y))`| **0.005**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`transpose(H)@H@x`| 0.525 | 165.705 | 0.997 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.527**| **166.276** | | |
|**Ref (+)**| `transpose(H)@(H@x)`| **0.003**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`transpose(y)@transpose(H)@H`| 0.003 | 0.25 | 0.001 | :white_check_mark: |  
|**Ref (-)** |`eval. right to left`| **0.526**| **229.57** | | |
|**Ref (+)**| `(transpose(y)@transpose(H))@H`| **0.002**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.54 | 9.065 | 0.999 | :x: | 
|**Ref (-)** |`eval. left to right`| **0.541**| **9.077** | | |
|**Ref (+)**| `(transpose(H)@y)@(transpose(x)@H)`| **0.054**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 1.045 | 0.927 | 1.001 | :x: |
|**Ref (-)** |`no rewrite`| **1.044**| **0.926** | | |
|**Ref (+)**|`A@(B+C)`|**0.542**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - transpose(H)@H)@x`| 0.528 | 138.794 | 1.0 | :x: |
|**Ref (-)** |`no rewrite`| **0.529**| **138.825** | | |
|**Ref (+)**|`A@x - transpose(H)@(H@x)`|**0.004**| - | | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks $A_1$ and $A_2$ along the diagnonal, each of size $ 1500 \times 1500 $, and the remaining elements are zero. The result of the matrix multiplication $AB$ can be rewritten as $[(A_1B_1), (A_2B_2)]$, where $B_1, B_2$ are of sizes $1500 \times 3000$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$AB$|`A@B`| 0.506 | 0.92 | 0.945 | :x: |
|$"$|`linalg.matmul(A,B)` | 0.519 | 0.969 | 0.996 | :x: |
|**Ref (-)** |`no rewrite`| **0.52**| **0.973** | | |
|**Ref (+)**|`blocked matrix multiply`|**0.264**| - | | |


## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.577 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.71**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.57**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.022 | 10.115 | 1.327 | :x: |
|**Ref (-)** |`no code motion`| **0.017**| **7.625** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.523 | 260.735 | 1.011 | :x: |
|**Ref (-)** |`no code motion`| **0.518**| **257.99** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.708 

### Score: 6 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />