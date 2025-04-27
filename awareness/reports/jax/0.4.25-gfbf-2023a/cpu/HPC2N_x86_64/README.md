## LAAB-Python | LA Awareness-CPU | jax/0.4.25-gfbf-2023a | HPC2N_x86_64

This report evaluates whether the software build performs operations equivalent to those of optimized math libraries (e.g., OpenBLAS, MKL), and whether it leverages linear algebra techniques to accelerate CPU computations.  Unless stated otherwise, all benchmarks use matrices of size $3000 \times 3000$ and are executed on a single CPU core of AMD EPYC 9454 48-Core Processor. 

### 1) Jax vs. BLAS for matrix multiplication:

TensorFlow's matrix multiplication - using the `@` operator and the `jax.numpy.matmul` function - is benchmarked. Given the matrices $A$ and $B$, the time taken for general matrix multiplication $A^TB$ is compared for equivalence against the reference `sgemm` routine invoked via OpenBLAS from C.

||Call  |  time (s)  | 
|----|------|------------|
|$A^TB$|`transpose(A)@B`| 0.4967 :white_check_mark:|
|$"$|`jax.numpy.matmul(t(A),B)` | 0.4968 :white_check_mark: |
|**Reference** |`sgemm`| **0.4623**|

<hr style="border: none; height: 1px; background-color: #ccc;" />

### 2) Elimination of common sub-expression:

 Sub-expressions within an expression that yield the same result can be computed once to eliminate redundant calculations.

  a) **Repeated in summation:**
  
  The input expression is $E = A^TB + A^TB$. The subexpression $A^TB$ appears twice. The execution time to evaluate $E$ is compared for equivalence against a reference implementation that computes $A^TB$ just once. 

|Expr |Call |time (s) |
|-----|-----|----------|
|$E$ |`transpose(A)@B + transpose(A)@B` | 1.1126 :white_check_mark:| 
|**Reference**| `2*(transpose(A)@B)`| **1.1198**|

  b) **Repeated in multiplication**

   Now, the benchmark is repeated with expressions $E_1 = (A^TB)^T(A^TB)$ and $E_2 = (A^TB)^TA^TB$. Evaluating these expressions from right to left involves three matrix multiplications. The reference implementation avoids the redundant computation of the common subexpression resulting in just two matrix multiplications to evaluate $E$.

|Expr|Call | time (s) |
|-----|-----|----------|
|$E_1$|`transpose(transpose(A)@B)@(transpose(A)@B)`| 1.6166 :x: |
|$E_2$|`transpose(transpose(A)@B)@transpose(A)@B`| 1.5448  :x: | 
|**Reference**| `S=transpose(A)@B; transpose(S)@S`| **1.1136**|


<hr style="border: none; height: 1px; background-color: #ccc;" />

### 3) Choosing the optimal order to evaluate a matrix chain:

Given $m$ matrices of suitable sizes, the product $M = A_1A_2...A_n$ is known as a matrix chain. Because of associativity of matrix product, matrix chain can be computed in many different ways, each identified by a specific paranthesization. The paranthesization that evaluates $M$ with the least number of floating point operations (FLOPs) is considered optimal.

  a) **Right to left**:

   The input matrix chain is $H^THx$, where $x$ is a vector of lenth $3000$. The reference implementation, evaluating from right-to-left - i.e.,  $H^T(Hx)$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$. 

|Expr|Call| time (s)|
|----|----|---------|
|$H^THx$|`transpose(H)@H@x`| 0.5566 :x: | 
|$"$|`linalg.multi_dot([transpose(H), H, x])`| 0.0121 :white_check_mark: | 
|**Reference**| `transpose(H)@(H@x)`| **0.0119**|

  b) **Left to right**:

  Now, the input matrix chain is $y^TH^TH$, where $y$ is a vector of lenth $3000$. The reference implementation, evaluating from left-to-right - i.e.,  $(y^TH^T)H$, avoids the expensive $\mathcal{O}(n^3)$ matrix product, and has a complexity of $\mathcal{O}(n^2)$.

|Expr|Call | time (s)|
|----|-----|---------|
|$y^TH^TH$|`transpose(y)@t(H)@H`| 0.0101 :white_check_mark: | 
|$"$|`linalg.multi_dot([transpose(y), transpose(H), H])`| 0.0101 :white_check_mark: | 
|**Reference**| `(transpose(y)@transpose(H))@H`| **0.0101**|


  c) **Mixed**:

  Now, the input matrix chain is $H^Tyx^TH$, where the vectors $x,y$ occur in the middle of a matrix chain. Here, neither left-to-right nor right-to-left evaluation avoids the expensive $\mathcal{O}(n^3)$ operation; instead, the evaluation $(H^Ty)(x^TH)$ turns out to be the optimum with $\mathcal{O}(n^2)$ complexity.  

|Expr|Call| time (s) |
|----|----|-----------|
|$H^Tyx^TH$|`transpose(H)@y@transpose(x)@H`| 0.5200 :x: | 
|$"$|`linalg.multi_dot([transpose(H), y, transpose(x), H])`| 0.0278 :white_check_mark: | 
|**Reference**| `(transpose(H)@y)@(transpose(x)@H)`| **0.0260**|

<hr style="border: none; height: 1px; background-color: #ccc;" />

### 4) Exploiting matrix properties:

 The input is a matrix multiplication $AB$ with matrix $A$ having certain special properties. Exploting the property of $A$ can result in accelerated evaluations.

  a) **TRMM**

  Matrix $A$ is lower triangular. The execution time for evaluation of the input expression is checked for equivalence against a reference implementation that utilizes the BLAS kernel `trmm`, which computes the matrix product with half the number of FLOPs than that required by `gemm`.

|Expr|Call |  time (s)  | 
|----|-----|------------|
|$AB$|`A@B`| 0.4939 :x: |
|$"$|`jax.numpy.matmul(A,B)`| 0.4969 :x:  |
|**Reference** |`trmm`| **0.2358**|

  b) **SYRK**

  Matrix $A$ is transpose of matrix B, and the resulting matrix $AA^T$ is symmetric. BLAS offers a specialized routine, `syrk` ("SYmmetric Rank-K update"), which computes the matrix product with only half the number of FLOPs than `gemm`. The reference implementation uses `syrk` for the matrix product. 

|Expr|Call| time (s)  | 
|----|----|------------|
|$AB$|`A@B`| 0.4883 :x: |
|$"$|`jax.numpy.matmul(A,B)`| 0.4932 :x:  |
|**Reference** |`syrk`| **0.2397**|

  c) **Tri-diagonal**

  Matrix A is tri-diagonal. The reference implementation performs the matrix multiplication using the compressed sparse row format for $A$, implemented in C, significantly reducing the number of FLOPs required for the computation. 

|Expr|Call|  time (s)  | 
|----|----|-------------|
|$AB$|`A@B`| 0.4926 :x: |
|$"$|`jax.numpy.matmul(A,B)`|    |
|**Reference** |`csr(A)@B in C`| **0.0045**|


<hr style="border: none; height: 1px; background-color: #ccc;" />

### 5) Algebraic Manipulations

 Algebraic properties can be used to rewrite an expression in several alternative ways that can result in accelerated evaluations.

  a) **Taking advantage of the distributive law**:

The input expression is $E_1 = AB+AC$. This expression requires two $\mathcal{O}(n^3)$ matrix multiplications.  $E_1$ can be rewritten using the distributive law as $A(B+C)$, reducing the number of $\mathcal{O}(n^3)$ matrix multiplications to one.

|Expr|Call| time (s)|
|----|---|----------|
|$E_1$|`A@B + A@C`| 1.0059 :x:| 
|**Reference**|`A@(B+C)`|**0.5111**|

Now, the input expression is $E_2 = (A - H^TH)x$, which involves one $\mathcal{O}(n^3)$ matrix multiplication. This expression can be rewritten as $Ax - H^T(Hx)$, thereby avoiding the $\mathcal{O}(n^3)$ matrix multiplcation. 

|Expr|Call| time (s)|
|----|---|----------|
|$E_2$|`(A - transpose(H)@H)@x`| 0.5646 :x:| 
|**Reference**|`A@x - transpose(H)@(H@x)`|**0.0136**|


  b) **Identifying the blocked matrix structure**:

  The matrix $A$ has a blocked structure as shown below. The reference implementation computes the matrix product $AB$ by evaluating only the product with individual blocks, which is less expensive than the multiplication with the large matrix. 
  
```math
A := \begin{bmatrix} A_1 & 0 \\ 0 & A_2 \end{bmatrix}
\qquad
B := \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}
\qquad
AB := \begin{bmatrix} (A_1B_1) \\ (A_2B_2) \end{bmatrix}
```

|Expr|Call| time (s)|
|----|---|----------|
|$AB$|`A@B`| 0.4899 :x: | 
|$"$|`jax.numpy.matmul(A,B)` |   | 
|**Reference**|`blocked matrix multiply`|**0.2732**|


<hr style="border: none; height: 1px; background-color: #ccc;" />

### 6) Code Motion

Some operations, when moved around, can result in improved performance.

  a) **Identifying the loop-invariant code:**

 Operations that occur within a loop but yield the same result regardless of how many times the loop is executed, can be computed just once and moved outside the loop.

||Call| time (s)|
|----|------|----------|
||`for i in range(3):` <br> `   A@B + tensordot(V[i],t(V[i])`| 0.5127  :white_check_mark: |
|**Reference**|`S=A@B;` <br> `for i in range(3):` <br>`   S+tensordot(V[i],t(V[i]) `|**0.5131**| 

  b) **Identifying partial operand access**:

The output of the expression `(A+B)[2,2]` requires only single element of both the matrices. Here, the explicit addition of the full matrices can be avoided. 

||Call | time (s)|
|----|-----|---------|
||`(A+B)[2,2]`| 0.0000 :x: | 
|**Reference**|`A[2]+B[2]`|**0.0000**|

Similarly, the output of the expression `(A@B)[2,2]` also requires only single element of both the matrices. Here, the explicit multiplication of the full matrices can be avoided. 

||Call | time (s)|
|----|-----|---------|
||`(A@B)[2,2]`| 0.4987 :x: | 
|**Reference**|`tensordot(A[2,:],B[:,2])`|**0.0000**|


<hr style="border: none; height: 1px; background-color: #ccc;" />

## Overall Score: 6/16
