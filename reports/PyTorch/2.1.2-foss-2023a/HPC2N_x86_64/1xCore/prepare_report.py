from laab_python.laab_results import LAABResults
from laab_python.prepare_report import prepare_markdown_report
import os
import sys

    
#get root dir from environ
template_file = "../../../../../templates/report_template.md"

laab_results = LAABResults("data.txt", "PyTorch/2.1.2-foss-2023a", "HPC2N_x86_64")

exp_config = {
    "laab_n":3000,
    "laab_rep":10,
    "name": "1xCPU-Core",
    "omp_num_threads": 1
}

prepare_markdown_report(laab_results, 'config.json', exp_config, template_file, "README.md", cutoff=0.05)
    