# Report | LAAB-Python | 1xCPU-Core 

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


||Call  |  time (s)  | loss | result@0.10 | 
|----|------|------------|--|---|
|$A^TB$|`t(A)@B`| 0.508 | 0.027| :white_check_mark: |
|$"$|`linalg.matmul(t(A),B)` | 0.512 | 0.033 | :white_check_mark: |
|**Reference** |`sgemm`| **0.495**| | |


## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | 0.523 | 0.0| :white_check_mark: | 
|**Reference**| `2*(t(A)@B)`| **0.523**| | |


b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| 1.02 | 0.0 | :white_check_mark: |
|**Reference**| `S=t(A)@B; t(S)@S`| **1.021**| | |

c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_3$|`t(t(A)@B)@t(A)@B`| 1.526 |  0.504 | :x: |
|**Reference**| `S=t(A)@B; t(S)@S`| **1.021**| | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$ and $y \in \mathbb{R}^{ 3000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | loss | result@0.10 |
|-----|-----|----------|--|--|
|$E_4$|`t(A)@B@t(A)@B@y`| 1.528 |  112.417 | :x: |
|**Reference**| `t(A)@(B@(t(A)@(B@y))`| **0.013**| | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $x \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| loss | result@0.10 |
|----|----|---------|--|--|
|$H^THx$|`t(H)@H@x`| 0.51 | 95.092 | :x: | 
|$"$|`linalg.multi_dot([t(H), H, x])`| 0.006 | 0.19 | :x: |  
|**Reference**| `t(H)@(H@x)`| **0.005**| | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$, $y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| loss | result@0.10 |
|----|-----|---------|--|--|
|$y^TH^TH$|`t(y)@t(H)@H`| 0.006 | 0.0 | :white_check_mark: |  
|$"$|`linalg.multi_dot([t(y), t(H), H])`| 0.006 | 0.0 | :white_check_mark: | 
|**Reference**| `(t(y)@t(H))@H`| **0.006**| | |

c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 3000 \times 3000 }$ and $x,y \in \mathbb{R}^{ 3000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | loss | result@0.10 |
|----|----|-----------|--|--|
|$H^Tyx^TH$|`t(H)@y@t(x)@H`| 0.527 | 20.589 | :x: | 
|$"$|`linalg.multi_dot([t(H), y, t(x), H])`| 0.025 | 0.027 | :white_check_mark: | 
|**Reference**| `(t(H)@y)@(t(x)@H)`| **0.024**| | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | loss | result@0.10|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.505 | 1.047 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.508 | 1.059  | :x: |
|**Reference** |`trmm`| **0.247**| | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | loss | result@0.10|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.506 | 1.002 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.509 | 1.015  | :x: |
|**Reference** |`syrk`| **0.253**| | |

c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | loss | result@0.10|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.508 | 114.684 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.511 | 115.46  | :x: | 
|**Reference** |`csr(A)@B`| **0.004**| | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
|$E_1$|`A@B+ A@C`| 1.042 | 0.965| :x: |
|**Reference**|`A@(B+C)`|**0.528**| | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
|$E_2$|`(A - t(H)@H)@x`| 0.527 | 53.393| :x: |
|**Reference**|`A@x - t(H)@(H@x)`|**0.01**| | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $ 1500 \times 1500 $.

|Expr|Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
|$AB$|`A@B`| 0.497 | 0.878 | :x: |
|$"$|`linalg.matmul(A,B)` | 0.509 | 0.876 | :x: |
|**Reference**|`blocked matrix multiply`|**0.264**| | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
||`for i in range(3): A@B ...`| 0.544 |  0.0 | :white_check_mark: |
|**Reference**|`A@B; for i in range(3): ...`|**0.545**| | | 

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
||`(A+B)[2,2]`| 0.016 | 6.639 | :x: |
|**Reference**|`A[2,2] + B[2,2]`|**0.002**| | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 3000 \times 3000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| loss | result@0.10 |
|----|---|----------|--|--|
||`(A@B)[2,2]`| 0.503 | 200.199 | :x: |
|**Reference**|`dot(A[2,:],B[:,2])`|**0.002**| | |


## OVERALL RESULT

### Mean loss: 28.939 

### Score: 6 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />