# Report | LAAB-Python | 1xSocket 

| Framework | PyTorch/2.1.2-foss-2023a | 
|---|---|
| **System** | HPC2N_x86_64 |
| **CPU** | Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz | 
| **LAAB_N** | 6000 |
| **LAAB_REP** | 10 |
| **OMP_NUM_THREADS** | 14 |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Comparison with GEMM

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | loss | result@0.05 | 
|----|------|------------|--|---|
|$A^TB$|`t(A)@B`| 0.288 | 0.239| :x: |
|$"$|`linalg.matmul(t(A),B)` | 0.292 | 0.259 | :x: |
|**Reference** |`sgemm`| **0.232**| | |


## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | loss | result@0.05 |
|-----|-----|----------|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | 0.303 | 0.0| :white_check_mark: | 
|**Reference**| `2*(t(A)@B)`| **0.304**| | |


b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@0.05 |
|-----|-----|----------|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| 0.581 | 0.0 | :white_check_mark: |
|**Reference**| `S=t(A)@B; t(S)@S`| **0.578**| | |

c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@0.05 |
|-----|-----|----------|--|--|
|$E_3$|`t(t(A)@B)@t(A)@B`| 0.877 |  0.504 | :x: |
|**Reference**| `S=t(A)@B; t(S)@S`| **0.578**| | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$ and $y \in \mathbb{R}^{ 6000 }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | loss | result@0.05 |
|-----|-----|----------|--|--|
|$E_4$|`t(A)@B@t(A)@B@y`| 0.863 |  75.195 | :x: |
|**Reference**| `t(A)@(B@(t(A)@(B@y))`| **0.011**| | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ 6000 \times 6000 }$, $x \in \mathbb{R}^{ 6000 }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| loss | result@0.05 |
|----|----|---------|--|--|
|$H^THx$|`t(H)@H@x`| 0.29 | 50.828 | :x: | 
|$"$|`linalg.multi_dot([t(H), H, x])`| 0.006 | 0.023 | :white_check_mark: |  
|**Reference**| `t(H)@(H@x)`| **0.006**| | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ 6000 \times 6000 }$, $y \in \mathbb{R}^{ 6000 }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| loss | result@0.05 |
|----|-----|---------|--|--|
|$y^TH^TH$|`t(y)@t(H)@H`| 0.006 | 0.0 | :white_check_mark: |  
|$"$|`linalg.multi_dot([t(y), t(H), H])`| 0.006 | 0.007 | :white_check_mark: | 
|**Reference**| `(t(y)@t(H))@H`| **0.006**| | |

c) **Mixed**:

Operands: $H \in \mathbb{R}^{ 6000 \times 6000 }$ and $x,y \in \mathbb{R}^{ 6000 }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | loss | result@0.05 |
|----|----|-----------|--|--|
|$H^Tyx^TH$|`t(H)@y@t(x)@H`| 0.347 | 4.266 | :x: | 
|$"$|`linalg.multi_dot([t(H), y, t(x), H])`| 0.066 | 0.0 | :white_check_mark: | 
|**Reference**| `(t(H)@y)@(t(x)@H)`| **0.066**| | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | loss | result@0.05|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.28 | 1.19 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.286 | 1.235  | :x: |
|**Reference** |`trmm`| **0.128**| | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | loss | result@0.05|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.283 | 0.0 | :white_check_mark: |
|$"$|`linalg.matmul(A,B)`| 0.288 | 0.0  | :white_check_mark: |
|**Reference** |`syrk`| **1.104**| | |

c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | loss | result@0.05|
|----|-----|------------|--|--|
|$AB$|`A@B`| 0.28 | 53.066 | :x: |
|$"$|`linalg.matmul(A,B)`| 0.288 | 54.569  | :x: |
|**Reference** |`csr(A)@B`| **0.005**| | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| loss | result@0.05 |
|----|---|----------|--|--|
|$E_1$|`A@B+ A@C`| 0.586 | 0.936| :x: |
|**Reference**|`A@(B+C)`|**0.303**| | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| loss | result@0.05 |
|----|---|----------|--|--|
|$E_2$|`(A - t(H)@H)@x`| 0.306 | 33.971| :x: |
|**Reference**|`A@x - t(H)@(H@x)`|**0.009**| | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $ 3000 \times 3000 $.

|Expr|Call| time (s)| loss | result@0.05 |
|----|---|----------|--|--|
|$AB$|`A@B`| 0.28 | 0.461 | :x: |
|$"$|`linalg.matmul(A,B)` | 0.28 | 0.463 | :x: |
|**Reference**|`blocked matrix multiply`|**0.192**| | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| loss | result@0.05 |
|----|---|----------|--|--|
||`for i in range(3): A@B ...`| 0.359 |  0.0 | :white_check_mark: |
|**Reference**|`A@B; for i in range(3): ...`|**0.358**| | | 

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| loss | result@0.05 |
|----|---|----------|--|--|
||`(A+B)[2,2]`| 0.009 | 0.614 | :x: |
|**Reference**|`A[2,2] + B[2,2]`|**0.006**| | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ 6000 \times 6000 }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| loss | result@0.05 |
|----|---|----------|--|--|
||`(A@B)[2,2]`| 0.28 | 44.438 | :x: |
|**Reference**|`dot(A[2,:],B[:,2])`|**0.006**| | |


## OVERALL RESULT

### Mean loss: 12.391 

### Score: 7 / 17

<hr style="border: none; height: 1px; background-color: #ccc;" />