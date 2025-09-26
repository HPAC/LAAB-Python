# Report | LAAB-Python | {{ exp_config.name }} 

| Framework | {{ eb_name }} | 
|---|---|
| **System** | {{ system }} |
| **CPU** | {{ cpu_model }} | 
| **LAAB_N** | {{ exp_config.laab_n }} |
| **LAAB_REP** | {{ exp_config.laab_rep }} |
| **OMP_NUM_THREADS** | {{ exp_config.omp_num_threads }} |

<!-- <hr style="border: none; height: 1px; background-color: #ccc;" /> -->


## Test 1: Matrix multiplications

The execution times of matrix multiplications invoked through the high-level APIs of the frameworks are compared against those of an optimised reference implementation.

a) **GEMM**:

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | slowdown | loss | result@{{ cutoff }} | 
|----|------|------------|--|---|--|
|$A^TB$|`{{ config.mm_sgemm.tests.operator }}`| {{ times.mm_sgemm.tests.operator }} | {{ slow_down.mm_sgemm.operator }} | {{ losses.mm_sgemm.operator }}| {{ cutoff_results.mm_sgemm.operator }} |
|$"$|`{{ config.mm_sgemm.tests.linalg_matmul }}` | {{ times.mm_sgemm.tests.linalg_matmul }} |  {{ slow_down.mm_sgemm.linalg_matmul }} | {{ losses.mm_sgemm.linalg_matmul }} | {{ cutoff_results.mm_sgemm.linalg_matmul }} |
|**Ref (-)** |`sgemv for each row`| **{{ times.mm_sgemm.ref_negative }}**| **{{ slow_down.mm_sgemm.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_sgemm.ref_positive }}`| **{{ times.mm_sgemm.ref_positive }}**| - | | |

b) **TRMM**

Operands: $L, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $LB$, where $T$ is lower triangular. The reference implementation utilises the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|--|
|$LB$|`{{ config.mm_trmm.tests.operator }}`| {{ times.mm_trmm.tests.operator }} | {{ slow_down.mm_trmm.operator }} | {{ losses.mm_trmm.operator }} | {{ cutoff_results.mm_trmm.operator }} |
|$"$|`{{ config.mm_trmm.tests.linalg_matmul }}`| {{ times.mm_trmm.tests.linalg_matmul }} |  {{ slow_down.mm_trmm.linalg_matmul }} | {{ losses.mm_trmm.linalg_matmul }}  | {{ cutoff_results.mm_trmm.linalg_matmul }} |
|**Ref (-)** |`sgemm`| **{{ times.mm_trmm.ref_negative }}**| **{{ slow_down.mm_trmm.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_trmm.ref_positive }}`| **{{ times.mm_trmm.ref_positive }}**| - | | |

c) **SYRK**

Operands: $A \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AA^T$. The reference implementation utilises the BLAS routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`.

|Expr|Call |  time (s)  | slowdown | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|--|
|$AA^{T}$|`{{ config.mm_syrk.tests.operator }}`| {{ times.mm_syrk.tests.operator }} | {{ slow_down.mm_syrk.operator }} | {{ losses.mm_syrk.operator }} | {{ cutoff_results.mm_syrk.operator }} |
|$"$|`{{ config.mm_syrk.tests.linalg_matmul }}`| {{ times.mm_syrk.tests.linalg_matmul }} | {{ slow_down.mm_syrk.linalg_matmul }} | {{ losses.mm_syrk.linalg_matmul }}  | {{ cutoff_results.mm_syrk.linalg_matmul }} |
|**Ref (-)** |`sgemm`| **{{ times.mm_syrk.ref_negative }}**| **{{ slow_down.mm_syrk.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_syrk.ref_positive }}`| **{{ times.mm_syrk.ref_positive }}**| - | | |


d) **Tri-diagonal**

Operands: $T, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $TB$, where $T$ is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $T$, implemented in C.

|Expr|Call |  time (s)  | slowdown | loss | result@{{ cutoff }}|
|----|-----|------------|--|--|--|
|$TB$|`{{ config.mm_tridiag.tests.operator }}`| {{ times.mm_tridiag.tests.operator }} | {{ slow_down.mm_tridiag.operator }} |{{ losses.mm_tridiag.operator }} | {{ cutoff_results.mm_tridiag.operator }} |
|$"$|`{{ config.mm_tridiag.tests.linalg_matmul }}`| {{ times.mm_tridiag.tests.linalg_matmul }} | {{ slow_down.mm_tridiag.linalg_matmul }} | {{ losses.mm_tridiag.linalg_matmul }}  | {{ cutoff_results.mm_tridiag.linalg_matmul }} | {% if config.mm_tridiag.tests.linalg_tridiagonal_matmul %}
|$"$|`{{ config.mm_tridiag.tests.linalg_tridiagonal_matmul }}`| {{ times.mm_tridiag.tests.linalg_tridiagonal_matmul }} | {{ slow_down.mm_tridiag.linalg_tridiagonal_matmul }} | {{ losses.mm_tridiag.linalg_tridiagonal_matmul }}  | {{ cutoff_results.mm_tridiag.linalg_tridiagonal_matmul }} | {% endif %}
|**Ref (-)** |`sgemm`| **{{ times.mm_tridiag.ref_negative }}**| **{{ slow_down.mm_tridiag.ref_negative }}** | | |
|**Ref (+)** |`{{ config.mm_tridiag.ref_positive }}`| **{{ times.mm_tridiag.ref_positive }}**| - | | |

## Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | slowdown |loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_1$ |`{{ config.cse_addition.tests.operator }}` | {{ times.cse_addition.tests.operator }} | {{ slow_down.cse_addition.operator }} | {{ losses.cse_addition.operator }}| {{ cutoff_results.cse_addition.operator }} |
|**Ref (-)** |`no cse`| **{{ times.cse_addition.ref_negative }}**| **{{ slow_down.cse_addition.ref_negative }}** | | | 
|**Ref (+)**| `{{ config.cse_addition.ref_positive }}`| **{{ times.cse_addition.ref_positive }}**| - | | |



b) **Repeated in multiplication (parenthesis)**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_2$|`{{ config.cse_matmul_paranthesis.tests.operator }}`| {{ times.cse_matmul_paranthesis.tests.operator }} | {{ slow_down.cse_matmul_paranthesis.operator }} | {{ losses.cse_matmul_paranthesis.operator }} | {{ cutoff_results.cse_matmul_paranthesis.operator }} |
|**Ref (-)** |`no cse`| **{{ times.cse_matmul_paranthesis.ref_negative }}**| **{{ slow_down.cse_matmul_paranthesis.ref_negative }}** | | |
|**Ref (+)**| `{{ config.cse_matmul_paranthesis.ref_positive }}`| **{{ times.cse_matmul_paranthesis.ref_positive }}**| - | | |


c) **Repeated in multiplication (no parenthesis)**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_3 = (A^TB)^TA^TB$. The reference implementation avoids the redundant computation of the common subexpression.

|Expr|Call | time (s) | slowdown | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_3$|`{{ config.cse_matmul_no_paranthesis.tests.operator }}`| {{ times.cse_matmul_no_paranthesis.tests.operator }} | {{ slow_down.cse_matmul_no_paranthesis.operator }} | {{ losses.cse_matmul_no_paranthesis.operator }} | {{ cutoff_results.cse_matmul_no_paranthesis.operator }} |
|**Ref (-)** |`no cse`| **{{ times.cse_matmul_no_paranthesis.ref_negative }}**| **{{ slow_down.cse_matmul_no_paranthesis.ref_negative }}** | | |
|**Ref (+)**| `{{ config.cse_matmul_no_paranthesis.ref_positive }}`| **{{ times.cse_matmul_no_paranthesis.ref_positive }}**| - | | |

d) **Sub-optimal CSE**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$ and $y \in \mathbb{R}^{ {{ exp_config.laab_n }} }$

Description: The input expression is $E_4 = A^TBA^TBy$. The reference implementation evaluates $E_4$ from right-to-left without CSE.

|Expr|Call | time (s) | slowdown | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|--|
|$E_4$|`{{ config.cse_suboptimal.tests.operator }}`| {{ times.cse_suboptimal.tests.operator }} | {{ slow_down.cse_suboptimal.operator }} | {{ losses.cse_suboptimal.operator }} | {{ cutoff_results.cse_suboptimal.operator }} |
|**Ref (-)** |`with cse`| **{{ times.cse_suboptimal.ref_negative }}**| **{{ slow_down.cse_suboptimal.ref_negative }}** | | |
|**Ref (+)**| `{{ config.cse_suboptimal.ref_positive }}`| **{{ times.cse_suboptimal.ref_positive }}**| - | | |

## Test 3: Matrix chains

a) **Right to left**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$, $x \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $H^THx$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|----|---------|--|--|--|
|$H^THx$|`{{ config.matchain_rtol.tests.operator }}`| {{ times.matchain_rtol.tests.operator }} | {{ slow_down.matchain_rtol.operator }} | {{ losses.matchain_rtol.operator }} | {{ cutoff_results.matchain_rtol.operator }} | {% if config.matchain_rtol.tests.linalg_multidot %}
|$"$|`{{ config.matchain_rtol.tests.linalg_multidot }}`| {{ times.matchain_rtol.tests.linalg_multidot }} | {{ slow_down.matchain_rtol.linalg_multidot }} | {{ losses.matchain_rtol.linalg_multidot }} | {{ cutoff_results.matchain_rtol.linalg_multidot }} |  {% endif %}
|**Ref (-)** |`eval. left to right`| **{{ times.matchain_rtol.ref_negative }}**| **{{ slow_down.matchain_rtol.ref_negative }}** | | |
|**Ref (+)**| `{{ config.matchain_rtol.ref_positive }}`| **{{ times.matchain_rtol.ref_positive }}**| - | | |

b) **Left to right**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$, $y \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $y^TH^TH$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)| slowdown | loss | result@{{ cutoff }} |
|----|-----|---------|--|--|--|
|$y^TH^TH$|`{{ config.matchain_ltor.tests.operator }}`| {{ times.matchain_ltor.tests.operator }} | {{ slow_down.matchain_ltor.operator }} | {{ losses.matchain_ltor.operator }} | {{ cutoff_results.matchain_ltor.operator }} |  {% if config.matchain_rtol.tests.linalg_multidot %}
|$"$|`{{ config.matchain_ltor.tests.linalg_multidot }}`| {{ times.matchain_ltor.tests.linalg_multidot }} | {{ slow_down.matchain_ltor.linalg_multidot }} | {{ losses.matchain_ltor.linalg_multidot }} | {{ cutoff_results.matchain_ltor.linalg_multidot }} | {% endif %}
|**Ref (-)** |`eval. right to left`| **{{ times.matchain_ltor.ref_negative }}**| **{{ slow_down.matchain_ltor.ref_negative }}** | | |
|**Ref (+)**| `{{ config.matchain_ltor.ref_positive }}`| **{{ times.matchain_ltor.ref_positive }}**| - | | |


c) **Mixed**:

Operands: $H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$ and $x,y \in \mathbb{R}^{ {{exp_config.laab_n}} }$

Description: The input matrix chain is $H^Tyx^TH$. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) | slowdown | loss | result@{{ cutoff }} |
|----|----|-----------|--|--|--|
|$H^Tyx^TH$|`{{ config.matchain_mixed.tests.operator }}`| {{ times.matchain_mixed.tests.operator }} | {{ slow_down.matchain_mixed.operator }} | {{ losses.matchain_mixed.operator }} | {{ cutoff_results.matchain_mixed.operator }} | {% if config.matchain_rtol.tests.linalg_multidot %}
|$"$|`{{ config.matchain_mixed.tests.linalg_multidot }}`| {{ times.matchain_mixed.tests.linalg_multidot }} | {{ slow_down.matchain_mixed.linalg_multidot }} | {{ losses.matchain_mixed.linalg_multidot }} | {{ cutoff_results.matchain_mixed.linalg_multidot }} | {% endif %}
|**Ref (-)** |`eval. left to right`| **{{ times.matchain_mixed.ref_negative }}**| **{{ slow_down.matchain_mixed.ref_negative }}** | | |
|**Ref (+)**| `{{ config.matchain_mixed.ref_positive }}`| **{{ times.matchain_mixed.ref_positive }}**| - | | |


## Test 4: Expression rewrites

a) **Distributivity 1**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
|$E_1$|`{{ config.am_distributivity1.tests.operator }}`| {{ times.am_distributivity1.tests.operator }} | {{ slow_down.am_distributivity1.operator }} | {{ losses.am_distributivity1.operator }} | {{ cutoff_results.am_distributivity1.operator }} |
|**Ref (-)** |`no rewrite`| **{{ times.am_distributivity1.ref_negative }}**| **{{ slow_down.am_distributivity1.ref_negative }}** | | |
|**Ref (+)**|`{{ config.am_distributivity1.ref_positive }}`|**{{ times.am_distributivity1.ref_positive }}**| - | | |

b) **Distributivity 2**

Operands: $A, H \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
|$E_2$|`{{ config.am_distributivity2.tests.operator }}`| {{ times.am_distributivity2.tests.operator }} | {{ slow_down.am_distributivity2.operator }} | {{ losses.am_distributivity2.operator }} | {{ cutoff_results.am_distributivity2.operator }} |
|**Ref (-)** |`no rewrite`| **{{ times.am_distributivity2.ref_negative }}**| **{{ slow_down.am_distributivity2.ref_negative }}** | | |
|**Ref (+)**|`{{ config.am_distributivity2.ref_positive }}`|**{{ times.am_distributivity2.ref_positive }}**| - | | |

c) **Transpose law**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $E_3 = B^TAA^TB$. This expression can be rewritten as $(A^TB)^T(A^TB)$ by applying the transpose law and the sub-expression $A^TB$ can be computed just once. 

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
|$E_3$|`{{ config.am_transpose.tests.operator }}`| {{ times.am_transpose.tests.operator }} | {{ slow_down.am_transpose.operator }} | {{ losses.am_transpose.operator }} | {{ cutoff_results.am_transpose.operator }} |
|**Ref (-)** |`no rewrite`| **{{ times.am_transpose.ref_negative }}**| **{{ slow_down.am_transpose.ref_negative }}** | | |
|**Ref (+)**|`{{ config.am_transpose.ref_positive }}`|**{{ times.am_transpose.ref_positive }}**| - | | |

d) **Blocked matrix**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$, where $A$ consists of two blocks $A_1$ and $A_2$ along the diagnonal, each of size $ {{ exp_config.laab_n//2 }} \times {{ exp_config.laab_n//2 }} $, and the remaining elements are zero. The result of the matrix multiplication $AB$ can be rewritten as $[(A_1B_1), (A_2B_2)]$, where $B_1, B_2$ are of sizes ${{ exp_config.laab_n//2 }} \times {{ exp_config.laab_n }}$. 

|Expr|Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
|$AB$|`{{ config.am_blocked.tests.operator }}`| {{ times.am_blocked.tests.operator }} | {{ slow_down.am_blocked.operator }} | {{ losses.am_blocked.operator }} | {{ cutoff_results.am_blocked.operator }} |
|$"$|`{{ config.am_blocked.tests.linalg_matmul }}` | {{ times.am_blocked.tests.linalg_matmul }} | {{ slow_down.am_blocked.linalg_matmul }} | {{ losses.am_blocked.linalg_matmul }} | {{ cutoff_results.am_blocked.linalg_matmul }} |
|**Ref (-)** |`no rewrite`| **{{ times.am_blocked.ref_negative }}**| **{{ slow_down.am_blocked.ref_negative }}** | | |
|**Ref (+)**|`{{ config.am_blocked.ref_positive }}`|**{{ times.am_blocked.ref_positive }}**| - | | |


## Test 5: Code motion

a) **Loop-invariant code motion**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $AB$ computed inside a loop. The reference implementation moves the repeated multiplication outside the loop.

||Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
||`{{ config.cm_loops.tests.operator }}`| {{ times.cm_loops.tests.operator }} | {{ slow_down.cm_loops.operator }} | {{ losses.cm_loops.operator }} | {{ cutoff_results.cm_loops.operator }} |
|**Ref (-)** |`no code motion`| **{{ times.cm_loops.ref_negative }}**| **{{ slow_down.cm_loops.ref_negative }}** | | |
|**Ref (+)**|`{{ config.cm_loops.ref_positive }}`|**{{ times.cm_loops.ref_positive }}**| - | | |

b) **Partial operand access in sum**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $(A+B)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit addition of the full matrices. 

||Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
||`{{ config.cm_partial_op_sum.tests.operator }}`| {{ times.cm_partial_op_sum.tests.operator }} | {{ slow_down.cm_partial_op_sum.operator }} | {{ losses.cm_partial_op_sum.operator }} | {{ cutoff_results.cm_partial_op_sum.operator }} |
|**Ref (-)** |`no code motion`| **{{ times.cm_partial_op_sum.ref_negative }}**| **{{ slow_down.cm_partial_op_sum.ref_negative }}** | | |
|**Ref (+)**|`{{ config.cm_partial_op_sum.ref_positive }}`|**{{ times.cm_partial_op_sum.ref_positive }}**| - | | |

c) **Partial operand access in product**

Operands: $A, B \in \mathbb{R}^{ {{exp_config.laab_n}} \times {{ exp_config.laab_n }} }$

Description: The input expression is $(AB)[2,2]$, which requires only single element of both the matrices. The  reference implementation avoids the explicit multiplication of the full matrices. 

||Call| time (s)| slowdown | loss | result@{{ cutoff }} |
|----|---|----------|--|--|--|
||`{{ config.cm_partial_op_prod.tests.operator }}`| {{ times.cm_partial_op_prod.tests.operator }} | {{ slow_down.cm_partial_op_prod.operator }} | {{ losses.cm_partial_op_prod.operator }} | {{ cutoff_results.cm_partial_op_prod.operator }} |
|**Ref (-)** |`no code motion`| **{{ times.cm_partial_op_prod.ref_negative }}**| **{{ slow_down.cm_partial_op_prod.ref_negative }}** | | |
|**Ref (+)**|`{{ config.cm_partial_op_prod.ref_positive }}`|**{{ times.cm_partial_op_prod.ref_positive }}**| - | | |


## OVERALL RESULT

### Mean loss: {{ mean_loss }} 

### Score: {{ score }} / {{ num_tests }}

<hr style="border: none; height: 1px; background-color: #ccc;" />
