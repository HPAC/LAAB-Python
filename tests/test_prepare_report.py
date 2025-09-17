from laab_python.prepare_report import prepare_markdown_report
from laab_python.laab_results import LAABResults

def test_pyt():
    template_file = "laab_python/templates/report_template.md"
    data_file = "tests/sample_data/data_pyt.txt"
    src_config_file = "laab_python/src/PyTorch/v2-cpu/config.json"
    eb_version = "PyTorch/2.1.2-foss-2023a"
    system = "HPC2N_x86_64"
    outfile = "tests/sample_data/report_pyt.md"
    
    exp_config = {
        "laab_n":3000,
        "laab_rep":10,
        "name": "1xCore",
        "omp_num_threads": 1
    }
    
    laab_results = LAABResults(data_file, eb_version, system)
    prepare_markdown_report(laab_results, src_config_file, exp_config, template_file, outfile, cutoff=0.1)
    # print(results)
   
def test_tf():
    template_file = "laab_python/templates/report_template.md"
    data_file = "tests/sample_data/data_tf.txt"
    src_config_file = "laab_python/src/TensorFlow/v2-cpu/config.json"
    eb_version = "TensorFlow/2.15.1-foss-2023a"
    system = "HPC2N_x86_64"
    outfile = "tests/sample_data/report_tf.md"
    
    exp_config = {
        "laab_n":3000,
        "laab_rep":10,
        "name": "1xCore",
        "omp_num_threads": 1
    }
    
    laab_results = LAABResults(data_file, eb_version, system)
    prepare_markdown_report(laab_results, src_config_file, exp_config, template_file, outfile, cutoff=0.1)

def test_jax():
    template_file = "laab_python/templates/report_template.md"
    data_file = "tests/sample_data/data_jax.txt"
    src_config_file = "laab_python/src/jax/v0-cpu/config.json"
    eb_version = "Jax/0.4.25-gfbf-2023a"
    system = "HPC2N_x86_64"
    outfile = "tests/sample_data/report_jax.md"
    
    exp_config = {
        "laab_n":3000,
        "laab_rep":10,
        "name": "1xCore",
        "omp_num_threads": 1
    }
    
    laab_results = LAABResults(data_file, eb_version, system)
    prepare_markdown_report(laab_results, src_config_file, exp_config, template_file, outfile, cutoff=0.1)
            
if __name__ == "__main__":
    test_pyt()
    test_tf()
    test_jax()