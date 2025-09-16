# Report | LAAB-Python | CPU 

| Framework | {{ eb_name }} | 
|---|---|
| **System** | {{ system }} |
| **CPU** | {{ cpu_model }} | 

<hr style="border: none; height: 1px; background-color: #ccc;" />


## Test 1: Comparison with GEMM

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | loss | result@{{ cutoff }} | 
|----|------|------------|--|---|
|$A^TB$|`t(A)@B`| {{ times.sgemm.tests.actual }} | {{ losses.sgemm.actual }}| {{ cutoff_results.sgemm.actual }} |
|$"$|`linalg.matmul(t(A),B)` | {{ times.sgemm.tests.linalg_matmul }} | {{ losses.sgemm.linalg_matmul }} | {{ cutoff_results.sgemm.linalg_matmul }} |
|**Reference** |`sgemm`| **{{ times.sgemm.optimized }}**| | |


## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | {{ times.cse_addition.tests.actual }} | {{ losses.cse_addition.actual }}| {{ cutoff_results.cse_addition.actual }} | 
|**Reference**| `2*(t(A)@B)`| **{{ times.cse_addition.optimized }}**| | |


b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| {{ times.cse_matmul_paranthesis.tests.actual }} | {{ losses.cse_matmul_paranthesis.actual }} | {{ cutoff_results.cse_matmul_paranthesis.actual }} |
|**Reference**| `S=t(A)@B; t(S)@S`| **{{ times.cse_matmul_paranthesis.optimized }}**| | |

c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_3$|`t(t(A)@B)@t(A)@B`| {{ times.cse_matmul_no_paranthesis.tests.actual }} |  {{ losses.cse_matmul_no_paranthesis.actual }} | {{ cutoff_results.cse_matmul_no_paranthesis.actual }} |
|**Reference**| `S=t(A)@B; t(S)@S`| **{{ times.cse_matmul_paranthesis.optimized }}**| | |

d) **Sub-optimal CSE**

TODO


## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{3000 \times 3000}$, $x \in \mathbb{R}^{3000}$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|----|---------|--|--|
|$H^THx$|`t(H)@H@x`| {{ times.matchain_rtol.tests.actual }} | {{ losses.matchain_rtol.actual }} | {{ cutoff_results.matchain_rtol.actual }} |
|$"$|`linalg.multi_dot([t(H), H, x])`| {{ times.matchain_rtol.tests.linalg_multidot }} | {{ losses.matchain_rtol.linalg_multidot }} | {{ cutoff_results.matchain_rtol.linalg_multidot }} |
|**Reference**| `t(H)@(H@x)`| **{{ times.matchain_rtol.optimized }}**| | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{3000 \times 3000}$, $y \in \mathbb{R}^{3000}$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| loss | result@{{ cutoff }} |
|----|-----|---------|--|--|
|$y^TH^TH$|`t(y)@t(H)@H`| {{ times.matchain_ltor.tests.actual }} | {{ losses.matchain_ltor.actual }} | {{ cutoff_results.matchain_ltor.actual }} |
|$"$|`linalg.multi_dot([t(y), t(H), H])`| {{ times.matchain_ltor.tests.linalg_multidot }} | {{ losses.matchain_ltor.linalg_multidot }} | {{ cutoff_results.matchain_ltor.linalg_multidot }} |
|**Reference**| `(t(y)@t(H))@H`| **{{ times.matchain_ltor.optimized }}**| | |

c) **Mixed**:

Operands: $H \in \mathbb{R}^{3000 \times 3000}$ and $x,y \in \mathbb{R}^{3000}$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | loss | result@{{ cutoff }} |
|----|----|-----------|--|--|
|$H^Tyx^TH$|`t(H)@y@t(x)@H`| {{ times.matchain_mixed.tests.actual }} | {{ losses.matchain_mixed.actual }} | {{ cutoff_results.matchain_mixed.actual }} |
|$"$|`linalg.multi_dot([t(H), y, t(x), H])`| {{ times.matchain_mixed.tests.linalg_multidot }} | {{ losses.matchain_mixed.linalg_multidot }} | {{ cutoff_results.matchain_mixed.linalg_multidot }} |
|**Reference**| `(t(H)@y)@(t(x)@H)`| **{{ times.matchain_mixed.optimized }}**| | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|
|$AB$|`A@B`| {{ times.mp_trmm.tests.actual }} | {{ losses.mp_trmm.actual }} | {{ cutoff_results.mp_trmm.actual }} |
|$"$|`linalg.matmul(A,B)`| {{ times.mp_trmm.tests.linalg_matmul }} | {{ losses.mp_trmm.linalg_matmul }}  | {{ cutoff_results.mp_trmm.linalg_matmul }} |
|**Reference** |`trmm`| **{{ times.mp_trmm.optimized }}**| | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|
|$AB$|`A@B`| {{ times.mp_syrk.tests.actual }} | {{ losses.mp_syrk.actual }} | {{ cutoff_results.mp_syrk.actual }} |
|$"$|`linalg.matmul(A,B)`| {{ times.mp_syrk.tests.linalg_matmul }} | {{ losses.mp_syrk.linalg_matmul }}  | {{ cutoff_results.mp_syrk.linalg_matmul }} |
|**Reference** |`syrk`| **{{ times.mp_syrk.optimized }}**| | |

c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|
|$AB$|`A@B`| {{ times.mp_tridiag.tests.actual }} | {{ losses.mp_tridiag.actual }} | {{ cutoff_results.mp_tridiag.actual }} |
|$"$|`linalg.matmul(A,B)`| {{ times.mp_tridiag.tests.linalg_matmul }} | {{ losses.mp_tridiag.linalg_matmul }}  | {{ cutoff_results.mp_tridiag.linalg_matmul }} |
|**Reference** |`csr(A)@B`| **{{ times.mp_tridiag.optimized }}**| | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
|$E_1$|`A@B + A@C`| {{ times.am_distributivity1.tests.actual }} | {{ losses.am_distributivity1.actual }}| {{ cutoff_results.am_distributivity1.actual }} |
|**Reference**|`A@(B+C)`|**{{ times.am_distributivity1.optimized }}**| | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
|$E_2$|`(A - t(H)@H)@x`| {{ times.am_distributivity2.tests.actual }} | {{ losses.am_distributivity2.actual }}| {{ cutoff_results.am_distributivity2.actual }} |
|**Reference**|`A@x - t(H)@(H@x)`|**{{ times.am_distributivity2.optimized }}**| | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $1500 \times 1500$.

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
|$AB$|`A@B`| {{ times.am_blocked.tests.actual }} | {{ losses.am_blocked.actual }} | {{ cutoff_results.am_blocked.actual }} |
|$"$|`linalg.matmul(A,B)` | {{ times.am_blocked.tests.linalg_matmul }} | {{ losses.am_blocked.linalg_matmul }} | {{ cutoff_results.am_blocked.linalg_matmul }} |
|**Reference**|`blocked matrix multiply`|**{{ times.am_blocked.optimized }}**| | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
||`for i in range(3):` <br> `   A@B + tensordot(V[i],t(V[i])`| {{ times.cm_loops.tests.actual }} |  {{ losses.cm_loops.actual }} | {{ cutoff_results.cm_loops.actual }} |
|**Reference**|`S=A@B;` <br> `for i in range(3):` <br>`   S+tensordot(V[i],t(V[i]) `|**{{ times.cm_loops.optimized }}**| | | 

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
||`(A+B)[2,2]`| {{ times.cm_partial_op_sum.tests.actual }} | {{ losses.cm_partial_op_sum.actual }} | {{ cutoff_results.cm_partial_op_sum.actual }} |
|**Reference**|`A[2]+B[2]`|**{{ times.cm_partial_op_sum.optimized }}**| | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
||`(A@B)[2,2]`| {{ times.cm_partial_op_prod.tests.actual }} | {{ losses.cm_partial_op_prod.actual }} | {{ cutoff_results.cm_partial_op_prod.actual }} |
|**Reference**|`dot(A[2,:],B[:,2])`|**{{ times.cm_partial_op_prod.optimized }}**| | |


## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
