from laab_python.laab_results import LAABResults
from laab_python.prepare_report import prepare_markdown_report, dump_results_pickle
import os
import sys
import subprocess
    
#get root dir from environ
git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
template_file = os.path.join(git_root, "laab_python/templates", "report_template.md")

laab_results = LAABResults("data.txt", "Jax/0.4.25-gfbf-2023a", "HPC2N_x86_64")

exp_config = {
    "laab_n":3000,
    "laab_rep":10,
    "name": "1xCPU-Core",
    "omp_num_threads": 1
}


pickle_path = "results.pkl"
dump_results_pickle(laab_results, 'config.json', exp_config, pickle_path, cutoff=0.05)

prepare_markdown_report(laab_results, 'config.json', exp_config, template_file, "README.md", cutoff=0.05)
    
