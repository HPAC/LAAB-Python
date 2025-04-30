## LAAB-Python | Performance | PyTorch/2.1.2-foss-2023a | HPC2N_x86_64


### 1) CPU performance

a) Weak scaling of `dgemm` over all CPU cores in a node. 

|System|Cores | Energy cap |  Performance (GFLOPs/s)| Avg. Node Power (Watt) | Energy (J/GFLOP)|
|------|----- |------------|------------------------|------------------------|-----------------|
|j|288|default|**7172**|1176.22|**0.16**|
|j|288|200 W |**8142**|1427|**0.18**|
|j|288|300 W |**8821**|1800|**0.20**|
|j|1|default |**868**|41|**21.16**|

b) Weak scaling of matrix multiplication using `@` or `linalg.matmul`

c) Weak scaling of matrix multiplication using `nn.sequential`
