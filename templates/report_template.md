# Report | LAAB-Python | {{ exp_config.name }} 

| Framework | {{ eb_name }} | 
|---|---|
| **System** | {{ system }} |
| **CPU** | {{ cpu_model }} | 
| **LAAB_N** | {{ exp_config.laab_n }} |
| **LAAB_REP** | {{ exp_config.laab_rep }} |
| **OMP_NUM_THREADS** | {{ exp_config.omp_num_threads }} |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Comparison with GEMM

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | loss | result@{{ cutoff }} | 
|----|------|------------|--|---|
|$A^TB$|`{{ config.sgemm.tests.actual }}`| {{ times.sgemm.tests.actual }} | {{ losses.sgemm.actual }}| {{ cutoff_results.sgemm.actual }} |
|$"$|`{{ config.sgemm.tests.linalg_matmul }}` | {{ times.sgemm.tests.linalg_matmul }} | {{ losses.sgemm.linalg_matmul }} | {{ cutoff_results.sgemm.linalg_matmul }} |
|**Reference** |`{{ config.sgemm.optimized }}`| **{{ times.sgemm.optimized }}**| | |


## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_1$ |`{{ config.cse_addition.tests.actual }}` | {{ times.cse_addition.tests.actual }} | {{ losses.cse_addition.actual }}| {{ cutoff_results.cse_addition.actual }} | 
|**Reference**| `{{ config.cse_addition.optimized }}`| **{{ times.cse_addition.optimized }}**| | |


b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_2$|`{{ config.cse_matmul_paranthesis.tests.actual }}`| {{ times.cse_matmul_paranthesis.tests.actual }} | {{ losses.cse_matmul_paranthesis.actual }} | {{ cutoff_results.cse_matmul_paranthesis.actual }} |
|**Reference**| `{{ config.cse_matmul_paranthesis.optimized }}`| **{{ times.cse_matmul_paranthesis.optimized }}**| | |

c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_3$|`{{ config.cse_matmul_no_paranthesis.tests.actual }}`| {{ times.cse_matmul_no_paranthesis.tests.actual }} |  {{ losses.cse_matmul_no_paranthesis.actual }} | {{ cutoff_results.cse_matmul_no_paranthesis.actual }} |
|**Reference**| `{{ config.cse_matmul_paranthesis.optimized }}`| **{{ times.cse_matmul_paranthesis.optimized }}**| | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$ and $y \in \mathbb{R}^{ {{ exp_config.laab_n }} }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_4$|`{{ config.cse_suboptimal.tests.actual }}`| {{ times.cse_suboptimal.tests.actual }} |  {{ losses.cse_suboptimal.actual }} | {{ cutoff_results.cse_suboptimal.actual }} |
|**Reference**| `{{ config.cse_suboptimal.optimized }}`| **{{ times.cse_suboptimal.optimized }}**| | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$, $x \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|----|---------|--|--|
|$H^THx$|`{{ config.matchain_rtol.tests.actual }}`| {{ times.matchain_rtol.tests.actual }} | {{ losses.matchain_rtol.actual }} | {{ cutoff_results.matchain_rtol.actual }} | {% if config.matchain_rtol.tests.linalg_multidot %}
|$"$|`{{ config.matchain_rtol.tests.linalg_multidot }}`| {{ times.matchain_rtol.tests.linalg_multidot }} | {{ losses.matchain_rtol.linalg_multidot }} | {{ cutoff_results.matchain_rtol.linalg_multidot }} |  {% endif %}
|**Reference**| `{{ config.matchain_rtol.optimized }}`| **{{ times.matchain_rtol.optimized }}**| | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$, $y \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| loss | result@{{ cutoff }} |
|----|-----|---------|--|--|
|$y^TH^TH$|`{{ config.matchain_ltor.tests.actual }}`| {{ times.matchain_ltor.tests.actual }} | {{ losses.matchain_ltor.actual }} | {{ cutoff_results.matchain_ltor.actual }} |  {% if config.matchain_rtol.tests.linalg_multidot %}
|$"$|`{{ config.matchain_ltor.tests.linalg_multidot }}`| {{ times.matchain_ltor.tests.linalg_multidot }} | {{ losses.matchain_ltor.linalg_multidot }} | {{ cutoff_results.matchain_ltor.linalg_multidot }} | {% endif %}
|**Reference**| `{{ config.matchain_ltor.optimized }}`| **{{ times.matchain_ltor.optimized }}**| | |

c) **Mixed**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$ and $x,y \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | loss | result@{{ cutoff }} |
|----|----|-----------|--|--|
|$H^Tyx^TH$|`{{ config.matchain_mixed.tests.actual }}`| {{ times.matchain_mixed.tests.actual }} | {{ losses.matchain_mixed.actual }} | {{ cutoff_results.matchain_mixed.actual }} | {% if config.matchain_rtol.tests.linalg_multidot %}
|$"$|`{{ config.matchain_mixed.tests.linalg_multidot }}`| {{ times.matchain_mixed.tests.linalg_multidot }} | {{ losses.matchain_mixed.linalg_multidot }} | {{ cutoff_results.matchain_mixed.linalg_multidot }} | {% endif %}
|**Reference**| `{{ config.matchain_mixed.optimized }}`| **{{ times.matchain_mixed.optimized }}**| | |


## Test 4: Matrix properties

a) **TRMM**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$, where $A$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|
|$AB$|`{{ config.mp_trmm.tests.actual }}`| {{ times.mp_trmm.tests.actual }} | {{ losses.mp_trmm.actual }} | {{ cutoff_results.mp_trmm.actual }} |
|$"$|`{{ config.mp_trmm.tests.linalg_matmul }}`| {{ times.mp_trmm.tests.linalg_matmul }} | {{ losses.mp_trmm.linalg_matmul }}  | {{ cutoff_results.mp_trmm.linalg_matmul }} |
|**Reference** |`{{ config.mp_trmm.optimized }}`| **{{ times.mp_trmm.optimized }}**| | |

b) **SYRK**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$, where $A$ is transpose of  $B$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|
|$AB$|`{{ config.mp_syrk.tests.actual }}`| {{ times.mp_syrk.tests.actual }} | {{ losses.mp_syrk.actual }} | {{ cutoff_results.mp_syrk.actual }} |
|$"$|`{{ config.mp_syrk.tests.linalg_matmul }}`| {{ times.mp_syrk.tests.linalg_matmul }} | {{ losses.mp_syrk.linalg_matmul }}  | {{ cutoff_results.mp_syrk.linalg_matmul }} |
|**Reference** |`{{ config.mp_syrk.optimized }}`| **{{ times.mp_syrk.optimized }}**| | |

c) **Tri-diagonal**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$, where $A$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C.

|Expr|Call |  time (s)  | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|
|$AB$|`{{ config.mp_tridiag.tests.actual }}`| {{ times.mp_tridiag.tests.actual }} | {{ losses.mp_tridiag.actual }} | {{ cutoff_results.mp_tridiag.actual }} |
|$"$|`{{ config.mp_tridiag.tests.linalg_matmul }}`| {{ times.mp_tridiag.tests.linalg_matmul }} | {{ losses.mp_tridiag.linalg_matmul }}  | {{ cutoff_results.mp_tridiag.linalg_matmul }} | {% if config.mp_tridiag.tests.linalg_tridiagonal_matmul %}
|$"$|`{{ config.mp_tridiag.tests.linalg_tridiagonal_matmul }}`| {{ times.mp_tridiag.tests.linalg_tridiagonal_matmul }} | {{ losses.mp_tridiag.linalg_tridiagonal_matmul }}  | {{ cutoff_results.mp_tridiag.linalg_tridiagonal_matmul }} | {% endif %}
|**Reference** |`{{ config.mp_tridiag.optimized }}`| **{{ times.mp_tridiag.optimized }}**| | |


## Test 5: Algebraic manipulations

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
|$E_1$|`{{ config.am_distributivity1.tests.actual }}`| {{ times.am_distributivity1.tests.actual }} | {{ losses.am_distributivity1.actual }}| {{ cutoff_results.am_distributivity1.actual }} |
|**Reference**|`{{ config.am_distributivity1.optimized }}`|**{{ times.am_distributivity1.optimized }}**| | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
|$E_2$|`{{ config.am_distributivity2.tests.actual }}`| {{ times.am_distributivity2.tests.actual }} | {{ losses.am_distributivity2.actual }}| {{ cutoff_results.am_distributivity2.actual }} |
|**Reference**|`{{ config.am_distributivity2.optimized }}`|**{{ times.am_distributivity2.optimized }}**| | |

c) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$, where $A$ consists of two blocks along the diagnonal, each of size $ {{ exp_config.laab_n//2 }} \times {{ exp_config.laab_n//2 }} $.

|Expr|Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
|$AB$|`{{ config.am_blocked.tests.actual }}`| {{ times.am_blocked.tests.actual }} | {{ losses.am_blocked.actual }} | {{ cutoff_results.am_blocked.actual }} |
|$"$|`{{ config.am_blocked.tests.linalg_matmul }}` | {{ times.am_blocked.tests.linalg_matmul }} | {{ losses.am_blocked.linalg_matmul }} | {{ cutoff_results.am_blocked.linalg_matmul }} |
|**Reference**|`{{ config.am_blocked.optimized }}`|**{{ times.am_blocked.optimized }}**| | |


## Test 6: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
||`{{ config.cm_loops.tests.actual }}`| {{ times.cm_loops.tests.actual }} |  {{ losses.cm_loops.actual }} | {{ cutoff_results.cm_loops.actual }} |
|**Reference**|`{{ config.cm_loops.optimized }}`|**{{ times.cm_loops.optimized }}**| | | 

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
||`{{ config.cm_partial_op_sum.tests.actual }}`| {{ times.cm_partial_op_sum.tests.actual }} | {{ losses.cm_partial_op_sum.actual }} | {{ cutoff_results.cm_partial_op_sum.actual }} |
|**Reference**|`{{ config.cm_partial_op_sum.optimized }}`|**{{ times.cm_partial_op_sum.optimized }}**| | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| loss | result@{{ cutoff }} |
|----|---|----------|--|--|
||`{{ config.cm_partial_op_prod.tests.actual }}`| {{ times.cm_partial_op_prod.tests.actual }} | {{ losses.cm_partial_op_prod.actual }} | {{ cutoff_results.cm_partial_op_prod.actual }} |
|**Reference**|`{{ config.cm_partial_op_prod.optimized }}`|**{{ times.cm_partial_op_prod.optimized }}**| | |


## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
