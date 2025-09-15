## LAAB-Python | CPU 

### Benchmark Information

|  {{ eb_name }} | |
|---|---|
| **System** | {{ system }} |
| **CPU** | {{ cpu_model }} | 


### Test 1: Comparison with GEMM

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.


||Call  |  time (s)  | loss | result@{{ cutoff }} | 
|----|------|------------|--|---|
|$A^TB$|`t(A)@B`| {{ times.sgemm.tests.actual }} | {{ losses.sgemm.actual }}| {{ cutoff_results.sgemm.actual }} |
|$"$|`linalg.matmul(t(A),B)` | {{ times.sgemm.tests.linalg_matmul }} | {{ losses.sgemm.linalg_matmul }} | {{ cutoff_results.sgemm.linalg_matmul }} |
|**Reference** |`sgemm`| **{{ times.sgemm.optimized }}**| | |

<hr style="border: none; height: 1px; background-color: #ccc;" />

### Test 2: CSE

a) **Repeated in summation:**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_1 = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_1$ |`t(A)@B + t(A)@B` | {{ times.cse_addition.tests.actual }} | {{ losses.cse_addition.actual }}| {{ cutoff_results.cse_addition.actual }} | 
|**Reference**| `2*(t(A)@B)`| **{{ times.cse_addition.optimized }}**| | |


b) **Repeated in multiplication**

Operands: $A, B \in \mathbb{R}^{3000 \times 3000}$

Description: The input expression is $E_2 = (A^TB)^T(A^TB)$ and $E_3 = (A^TB)^TA^TB$. Evaluating these expressions from right to left involves three matrix multiplications. The reference implementation avoids the redundant computation of the common subexpression resulting in just two matrix multiplications.

|Expr|Call | time (s) | loss | result@{{ cutoff }} |
|-----|-----|----------|--|--|
|$E_2$|`t(t(A)@B)@(t(A)@B)`| {{ times.cse_matmul_paranthesis.tests.actual }} | {{ losses.cse_matmul_paranthesis.actual }} | {{ cutoff_results.cse_matmul_paranthesis.actual }} |
|$E_3$|`t(t(A)@B)@t(A)@B`| {{ times.cse_matmul_no_paranthesis.tests.actual }} |  {{ losses.cse_matmul_no_paranthesis.actual }} | {{ cutoff_results.cse_matmul_no_paranthesis.actual }} |
|**Reference**| `S=t(A)@B; t(S)@S`| **{{ times.cse_matmul_paranthesis.optimized }}**| | |

c) **Sub-optimal CSE**

TODO

<hr style="border: none; height: 1px; background-color: #ccc;" />

### Test 3: Matrix chains

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

<hr style="border: none; height: 1px; background-color: #ccc;" />