from laab_python.laab_results import LAABResults
from laab_python.prepare_report import prepare_markdown_report
import os
import sys
import subprocess
    
git_root = repo_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
template_file = os.path.join(git_root, "laab_python/templates", "report_template.md")

laab_results = LAABResults("data.txt", "PyTorch/2.1.2-foss-2023a", "HPC2N_x86_64")

exp_config = {
    "laab_n":6000,
    "laab_rep":10,
    "name": "1xSocket",
    "omp_num_threads": 14
}

prepare_markdown_report(laab_results, 'config.json', exp_config, template_file, "README.md", cutoff=0.05)
    