# Report | LAAB-Python | 1xCore 

| Framework | PyTorch/2.1.2-foss-2023a | 
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


||Call  |  time (s)  | slowdown | loss | result@0.10 | 
|----|------|------------|--|---|--|
|$A^TB$|`t(A)@B`| 0.508 | 0.02 | 0.006| :white_check_mark: |
|$"$|`linalg.matmul(t(A),B)` | 0.508 |  0.021 | 0.006 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **2.225**| **3.469** | | |
|**Ref (+)** |`sgemm`| **0.498**| - | | |



## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.10 |
|-----|-----|----------|--|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | 0.528 | 0.0 | 0.0| :white_check_mark: |
|**Ref (-)** |`no cse`| **1.054**| **1.0** | | | 
|**Ref (+)**| `2*(t(A)@B)`| **0.527**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.10 |
|-----|-----|----------|--|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| 1.014 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no cse`| **1.523**| **0.5** | | |
|**Ref (+)**| `S=t(A)@B; t(S)@S`| **1.015**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.10 |
|-----|-----|----------|--|--|--|
|$E_3$|`t(t(A)@B)@t(A)@B`| 1.532 | 0.502 | 1.003 | :x: |
|**Ref (-)** |`no cse`| **1.53**| **0.5** | | |
|**Ref (+)**| `S=t(A)@B; t(S)@S`| **1.02**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.10 |
|-----|-----|----------|--|--|--|
|$E_4$|`t(A)@B@t(A)@B@y`| 1.53 | 120.313 | 1.505 | :x: |
|**Ref (-)** |`with cse`| **1.021**| **79.954** | | |
|**Ref (+)**| `t(A)@(B@(t(A)@(B@y))`| **0.013**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.10 |
|----|----|---------|--|--|--|
|$H^THx$|`t(H)@H@x`| 0.512 | 96.809 | 1.001 | :x: | 
|$"$|`linalg.multi_dot([t(H), H, x])`| 0.006 | 0.136 | 0.001 | :white_check_mark: |  
|**Ref (-)** |`eval. left to right`| **0.511**| **96.751** | | |
|**Ref (+)**| `t(H)@(H@x)`| **0.005**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.10 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`t(y)@t(H)@H`| 0.006 | 0.092 | 0.001 | :white_check_mark: |  
|$"$|`linalg.multi_dot([t(y), t(H), H])`| 0.005 | 0.0 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. right to left`| **0.508**| **96.717** | | |
|**Ref (+)**| `(t(y)@t(H))@H`| **0.005**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.10 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`t(H)@y@t(x)@H`| 0.536 | 21.112 | 1.001 | :x: | 
|$"$|`linalg.multi_dot([t(H), y, t(x), H])`| 0.025 | 0.019 | 0.001 | :white_check_mark: | 
|**Ref (-)** |`eval. left to right`| **0.536**| **21.086** | | |
|**Ref (+)**| `(t(H)@y)@(t(x)@H)`| **0.024**| - | | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.10|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.511 | 1.062 | 1.054 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.511 |  1.062 | 1.054  | :x: |
|**Ref (-)** |`sgemm`| **0.498**| **1.008** | | |
|**Ref (+)** |`trmm`| **0.248**| - | | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.10|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.509 | 0.995 | 1.045 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.509 | 0.996 | 1.045  | :x: |
|**Ref (-)** |`sgemm`| **0.498**| **0.953** | | |
|**Ref (+)** |`syrk`| **0.255**| - | | |


c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.10|
|----|-----|------------|--|--|--|
|$AB$|`A@B`| 0.506 | 114.535 |1.017 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.506 | 114.572 | 1.017  | :x: | 
|**Ref (-)** |`sgemm`| **0.498**| **112.673** | | |
|**Ref (+)** |`csr(A)@B`| **0.004**| - | | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.10 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 1.037 | 0.965 | 0.993 | :x: |
|**Ref (-)** |`no rewrite`| **1.041**| **0.971** | | |
|**Ref (+)**|`A@(B+C)`|**0.528**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.10 |
|----|---|----------|--|--|--|
|$E_2$|`(A - t(H)@H)@x`| 0.529 | 56.697 | 1.0 | :x: |
|**Ref (-)** |`no rewrite`| **0.529**| **56.689** | | |
|**Ref (+)**|`A@x - t(H)@(H@x)`|**0.009**| - | | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $ 1500 \times 1500 $.

|Expr|Call| time (s)| slowdown | loss | result@0.10 |
|----|---|----------|--|--|--|
|$AB$|`A@B`| 0.505 | 0.899 | 1.019 | :x: |
|$"$|`linalg.matmul(A,B)` | 0.508 | 0.91 | 1.031 | :x: |
|**Ref (-)** |`no rewrite`| **0.5**| **0.882** | | |
|**Ref (+)**|`blocked matrix multiply`|**0.266**| - | | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.10 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.548 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.641**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.547**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.10 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.018 | 7.142 | 1.229 | :x: |
|**Ref (-)** |`no code motion`| **0.015**| **5.81** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.10 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.507 | 191.102 | 1.005 | :x: |
|**Ref (-)** |`no code motion`| **0.504**| **190.095** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.003**| - | | |


## OVERALL RESULT

### Mean loss: 0.640 

### Score: 7 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />