# Report | LAAB-Python | 1xCPU-Core 

| Framework | PyTorch/2.1.2-foss-2023a | 
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
|$A^TB$|`t(A)@B`| 0.509 | 0.023 | 0.007| :white_check_mark: |
|$"$|`linalg.matmul(t(A),B)` | 0.509 |  0.023 | 0.006 | :white_check_mark: |
|**Ref (-)** |`sgemv for each row`| **2.24**| **3.499** | | |
|**Ref (+)** |`sgemm`| **0.498**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$LB$|`L@B`| 0.509 | 1.053 | 1.045 | :x: |
|$"$|`linalg.matmul(L,B)`| 0.509 |  1.054 | 1.045  | :x: |
|**Ref (-)** |`sgemm`| **0.498**| **1.008** | | |
|**Ref (+)** |`trmm`| **0.248**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$AA^{T}$|`A@t(A)`| 0.508 | 1.009 | 1.042 | :x: |
|$"$|`linalg.matmul(A,t(A))`| 0.508 | 1.008 | 1.04  | :x: |
|**Ref (-)** |`sgemm`| **0.498**| **0.968** | | |
|**Ref (+)** |`syrk`| **0.253**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@0.05|
|----|-----|------------|--|--|--|
|$TB$|`T@B`| 0.507 | 115.101 |1.018 | :x: |
|$"$|`linalg.matmul(T,B)`| 0.507 | 115.092 | 1.018  | :x: | 
|**Ref (-)** |`sgemm`| **0.498**| **113.089** | | |
|**Ref (+)** |`csr(T)@B`| **0.004**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | 0.527 | 0.0 | 0.0| :white_check_mark: |
|**Ref (-)** |`no cse`| **1.054**| **1.0** | | | 
|**Ref (+)**| `2*(t(A)@B)`| **0.527**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| 1.013 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no cse`| **1.519**| **0.5** | | |
|**Ref (+)**| `S=t(A)@B; t(S)@S`| **1.013**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_3$|`t(t(A)@B)@t(A)@B`| 1.529 | 0.502 | 1.005 | :x: |
|**Ref (-)** |`no cse`| **1.527**| **0.5** | | |
|**Ref (+)**| `S=t(A)@B; t(S)@S`| **1.018**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@0.05 |
|-----|-----|----------|--|--|--|
|$E_4$|`t(A)@B@t(A)@B@y`| 1.531 | 110.408 | 1.507 | :x: |
|**Ref (-)** |`with cse`| **1.021**| **73.284** | | |
|**Ref (+)**| `t(A)@(B@(t(A)@(B@y))`| **0.014**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|----|---------|--|--|--|
|$H^THx$|`t(H)@H@x`| 0.51 | 90.487 | 1.0 | :x: | 
|$"$|`linalg.multi_dot([t(H), H, x])`| 0.006 | 0.099 | 0.001 | :white_check_mark: |  
|**Ref (-)** |`eval. left to right`| **0.51**| **90.471** | | |
|**Ref (+)**| `t(H)@(H@x)`| **0.006**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@0.05 |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`t(y)@t(H)@H`| 0.006 | 0.127 | 0.001 | :white_check_mark: |  
|$"$|`linalg.multi_dot([t(y), t(H), H])`| 0.006 | 0.005 | 0.0 | :white_check_mark: | 
|**Ref (-)** |`eval. right to left`| **0.509**| **90.226** | | |
|**Ref (+)**| `(t(y)@t(H))@H`| **0.006**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@0.05 |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`t(H)@y@t(x)@H`| 0.532 | 19.901 | 1.002 | :x: | 
|$"$|`linalg.multi_dot([t(H), y, t(x), H])`| 0.026 | 0.029 | 0.001 | :white_check_mark: | 
|**Ref (-)** |`eval. left to right`| **0.531**| **19.864** | | |
|**Ref (+)**| `(t(H)@y)@(t(x)@H)`| **0.025**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_1$|`A@B+ A@C`| 1.038 | 0.969 | 1.0 | :x: |
|**Ref (-)** |`no rewrite`| **1.038**| **0.969** | | |
|**Ref (+)**|`A@(B+C)`|**0.527**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$E_2$|`(A - t(H)@H)@x`| 0.531 | 55.043 | 1.0 | :x: |
|**Ref (-)** |`no rewrite`| **0.531**| **55.058** | | |
|**Ref (+)**|`A@x - t(H)@(H@x)`|**0.009**| - | | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks $A_1$ and $A_2$ along the diagnonal, each of size $ 1500 \times 1500 $, and the remaining elements are zero. The result of the matrix multiplication $AB$ can be rewritten as $[(A_1B_1), (A_2B_2)]$, where $B_1, B_2$ are of sizes $1500 \times 3000$. 

|Expr|Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
|$AB$|`A@B`| 0.507 | 0.905 | 1.051 | :x: |
|$"$|`linalg.matmul(A,B)` | 0.496 | 0.862 | 1.001 | :x: |
|**Ref (-)** |`no rewrite`| **0.496**| **0.861** | | |
|**Ref (+)**|`blocked matrix multiply`|**0.266**| - | | |


## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`for i in range(3): A@B ...`| 0.552 | 0.0 | 0.0 | :white_check_mark: |
|**Ref (-)** |`no code motion`| **1.656**| **2.0** | | |
|**Ref (+)**|`A@B; for i in range(3): ...`|**0.552**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A+B)[2,2]`| 0.019 | 8.375 | 1.139 | :x: |
|**Ref (-)** |`no code motion`| **0.017**| **7.356** | | |
|**Ref (+)**|`A[2,2] + B[2,2]`|**0.002**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@0.05 |
|----|---|----------|--|--|--|
||`(A@B)[2,2]`| 0.51 | 222.531 | 1.005 | :x: |
|**Ref (-)** |`no code motion`| **0.507**| **221.461** | | |
|**Ref (+)**|`dot(A[2,:],B[:,2])`|**0.002**| - | | |


## OVERALL RESULT

### Mean loss: 0.633 

### Score: 7 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />