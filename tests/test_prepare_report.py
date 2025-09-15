from laab_python.prepare_report import prepare_markdown_report
from laab_python.laab_results import LAABResults

def test1():
    template_file = "tests/sample_data/report_template.md"
    data_file = "tests/sample_data/data.txt"
    eb_version = "PyTorch/2.1.2-foss-2023a"
    system = "HPC2N_x86_64"
    outfile = "tests/sample_data/report.md"
    
    laab_results = LAABResults(data_file, eb_version, system)
    prepare_markdown_report(laab_results, template_file, outfile, cutoff=0.050)
    # print(results)
    
if __name__ == "__main__":
    test1()