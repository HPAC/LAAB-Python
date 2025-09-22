# LAAB - Linear Algebra Awareness Benchmarks for Python

Machine learning frameworks such as PyTorch, TensorFlow, and JAX are designed to scale numerical computations and data processing from single to multi-GPU and distributed computing environments, while abstracting much of the underlying complexity of parallelisation. Amid the growing use of machine learning in traditional high-performance computing application areas-for example, protein modeling and bioinformatics—researchers and practitioners increasingly employ these frameworks to integrate ML algorithms into their scientific computing applications, such as those involving solution of partial differential equations, eigenvalue problems, etc.

Linear algebra operations, which are the building blocks of countless computational problems, often constitute major performance bottlenecks. The HPC community invests significant effort in developing architecture-specific optimised kernels, such as those provided by the BLAS and LAPACK libraries, to accelerate these operations. Thanks to high-level frameworks, users no longer need to engage in the error-prone and time-consuming process of directly invoking such kernels; instead, they can express computations using a high-level, mathematics-like syntax that closely mirrors textbook notation, without having to manage the low-level implementation details themselves. However, it is not guaranteed that these frameworks will automatically exploit knowledge of linear algebra to accelerate computations in the way expert-tuned implementations would.

## The Benchmark suite

This benchmark consists of five tests that evaluate the linear algebra awareness of high-level Python interfaces in a given installation of PyTorch, TensorFlow, and JAX.

**Test 1: Matrix multiplications.** The performance of matrix multiplications invoked through the high-level APIs of the frameworks is compared against an optimised reference implementation. For general dense matrices, performance is compared with the BLAS kernel `gemm`. For triangular matrices and symmetric rank-k updates, which require only about half the floating-point operations of `gemm`, performance is compared against the specialised BLAS kernels `trmm` and `syrk`. For matrix multiplication with triangular matrices, the performance is compared a reference that uses compressed row formats. 


**Test 2: Common sub-expression elimination.** In general, sub-expressions that evaluate to the same value can be computed once and reused via a temporary reference in subsequent instances. This can improve performance in some cases, while in others it may not. This test evaluates the application of common sub-expression elimination on several input expressions.

**Test 3: Matrix chains.** Given $m$ matrices of suitable sizes, the product $M = A_1A_2...A_n$ is known as a matrix chain. Because of associativity of matrix product, matrix chain can be computed in many different ways, each identified by a specific paranthesisation. The paranthesisation that evaluates $M$ with the least number of floating point operations (FLOPs) is considered optimal. This test evaluates the performance of a set of input matrix chain expressions against reference implementations that use the optimal parenthesisation.

**Test 4: Expression rewrites.** Algebraic properties can be used to rewrite an expression in several alternative ways that can result in accelerated evaluations. This test evaluates rewrites based on the distributive law as well as transformations that break down blocked matrix multiplications.

**Test 5: Code motion.** Some operations, when moved around, can result in improved performance. This test evaluates if the frameworks can identify loop-invariant code and partial operand accesses to improve performance.


## Evaluation procedure

Each test consists of one or more input expressions in a high level syntax. The execution times of each input expression (measured 10 times) are compared against those of an optimised reference implementation and the fraction by which the input expression is slower than the reference implmentation is computed as follows,
$$
\texttt{slowdown}(\mathbf{t}, \mathbf{t}^{+}) =
    \begin{cases}
        \frac{\texttt{median}(\mathbf{t}) - \texttt{median}(\mathbf{t}^{+})}{\texttt{median}(\mathbf{t}')} & \text{if } \ \texttt{quantile}(\mathbf{t}^{+}, 75) \le \texttt{quantile}(\mathbf{t}, 25)  \\
        0.0 & \text{otherwise} 
    \end{cases} 
$$
Here, $\mathbf{t}$ and $\mathbf{t}^{+}$ are the list of execution time measurements of the input expression and the reference implementation respectively, and $\texttt{quantile}(\mathbf{t}, x)$ denote the $x^{th}$ quantile value of $\mathbf{t}$. For each input expression, we also consider a negative reference, which is a sub-optimal implmentation, and measure the execution times $\mathbf{t}^{-}$. Then, the loss is defined as the ratio of the slowdown of $\mathbf{t}$ to the slowdown of $\mathbf{t}^-$ with respect to the reference $\mathbf{t}^+$. That is,
$$
\texttt{loss}(\mathbf{t}, \mathbf{t}^+, \mathbf{t}^-) = \frac{\texttt{slowdown}(\mathbf{t}, \mathbf{t}^+)}{\texttt{slowdown}(\mathbf{t}^-, \mathbf{t}^+)}
$$
For each test, the $\texttt{slowdown}$ and $\texttt{loss}$ scores are reported. A test is considered pass if $\texttt{loss} \le 0.05$. For a given framework installation, the overall benchmark result include the mean $\texttt{loss}$ and the number of tests passed.

## Example and references

An example evaluation report for the installation of PyTorch/2.1.2-foss-2023a on the HPC system Kebnekaise, hosted at Umea University in Sweden is available [here](examples/PyTorch/2.1.2-foss-2023a/HPC2N_x86_64/1xCore/amd_zen3/).

For more details, refer to the following published articles:

```
Benchmarking the Linear Algebra Awareness of TensorFlow and PyTorch by Sankaran et al., published in the Proceedings of the 2022 IEEE International Parallel and Distributed
Processing Symposium Workshops (IPDPSW), Lyon, France, 2022, pp. 924-933. DOI: 10.1109/IPDPSW55747.2022.00150
```
