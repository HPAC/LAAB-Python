## Usage

The workflow consists of two stages:  
1. Executing the benchmark to obtain execution times for the test and reference expressions.  
2. Preparing the report and calculating the metrics from the collected performance data.  

### 1) Executing the benchmark

The execution of this benchmark requires the following:

1) Installation of the high-level framework to be tested (PyTorch, TensorFlow or JAX).
2) Installation of a math kernel library (e.g., OpenBLAS, MKL) that provides the `CBLAS` interface. The corresponding header files should be available in the include path, and the linker flags can be specified via the environment variable `LD_BLAS`.

The benchmark code corresponding to a given major version of the framework is located in the `src` directory. 
To execute a benchmark, navigate to the appropriate subdirectory and run the `make` command. The benchmark executes multiple BLAS reference implementations and test expressions, repeating each several times, and prints the results to standard output. By default, each test is repeated three times, and the matrix dimension (problem size) is set to 3000, with computations carried out in single precision. The number of repetitions and the problem size can be modified by setting the environment variables `LAAB_REPS` and `LAAB_N`, respectively. Set the environment variable `LD_BLAS` if necessary. Each test uses a single OpenMP thread by default; this can be changed by setting the environment variable `OMP_NUM_THREADS`.

**Example**:

An example usage of this benchmark to test PyTorch version 2.x installed on a HPC cluster is shown below. The measurements are written to data.txt.

```bash
module load GCC 
module load FlexiBLAS
module load PyTorch

export LD_BLAS='-lopenblas'
export OMP_NUM_THREADS=1
export LAAB_REPS=10
export LAAB_N=3000

# get the root directory of this git repo
git_root=$(git rev-parse --show-toplevel)
export SRC_DIR=$git_root/laab_python/src/PyTorch/v2-cpu/
make --no-print-directory -C $SRC_DIR/  | tee -a data.txt
```

### 2) Preparing the Report

To generate the report from a measurement file, you must first install the Python package provided in this repository:

```bash
pip install git+https://github.com/HPAC/LAAB-Python.git
```
To prepare the report, you need a template file from `laab_python/templates` (currently provided as a Markdown template) and the benchmark configuration file (`config.json`) located in the corresponding benchmark source directory. Once these are available, the report can be generated as follows:

```python
from laab_python.laab_results import LAABResults
from laab_python.prepare_report import prepare_markdown_report

laab_results = LAABResults("data.txt", eb_version="PyTorch/2.1.2-foss-2023a", system="HPC2N_x86_64")

template_file = os.path.join(git_root, "laab_python/templates", "report_template.md")
benchmark_config = "laab_python/src/PyTorch/v2-cpu/config.json"

exp_config = {
    "name": "1xCPU-Core",
    "laab_n":3000,
    "laab_rep":10,
    "omp_num_threads": 1  
}

outfile="README.md" # The name of the generated report file
prepare_markdown_report(laab_results, benchmark_config, exp_config, template_file, outfile)
```

**Example**:

An example generated report is available [here](PyTorch/2.1.2-foss-2023a/HPC2N_x86_64/1xCore/amd_zen3/README.md).

## Recommended format to store reports

For software installations on HPC systems, the following directory structure is recommended for storing reports:

```
reports/<FrameworkName>/<VersionString>/<SystemName>/<NumCores>/<Processor>/
```

Whenever possible, include the corresponding submit script and the raw data file in the same location.

**Example**:

The path `reports/PyTorch/2.1.2-foss-2023a/HPC2N_x86_64/1xCore/amd_zen3/` corresponds to an EasyBuild installation of PyTorch version `2.1.2-foss-2023a` on the `HPC2N_x86_64` system, executed on a single core of an `amd_zen3` CPU.

