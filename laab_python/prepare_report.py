import os
import sys
import statistics
from jinja2 import Template
import json
from .laab_results import LAABResults

def format_floats_recursive(data: dict, precision: int = 2) -> dict:
    """
    Recursively traverses a dictionary and formats all float values
    to a specified number of decimal places.
    """
    for key, value in data.items():
        if isinstance(value, float):
            # Format the float to the given precision
            data[key] = float(f"{value:.{precision}f}")
        elif isinstance(value, dict):
            # If the value is a dict, recurse into it
            format_floats_recursive(value, precision)
    return data

def format_cutoff_results_md(cutoff_results):
    for exp, tests in cutoff_results.items():
        for test, result in tests.items():
            if result == True:
                cutoff_results[exp][test] = ":white_check_mark:"
            else:
                cutoff_results[exp][test] = ":x:"
    return cutoff_results

def prepare_markdown_report(laab_results, src_config_file, exp_config, template_file, outfile, cutoff=0.05):
    
    if not os.path.exists(src_config_file):
        raise FileNotFoundError(f"Config file {src_config_file} not found")
    config = json.load(open(src_config_file, "r"))

    
    if set(laab_results.data.keys()) != set(config.keys()):
        raise ValueError("Experiments in data file and config file do not match")
    
    
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file {template_file} not found")
    
    min_exec_times = laab_results.get_min_test_times()
    laab_results.compute_loss()
    
    losses = laab_results.loss
    mean_loss = laab_results.mean_loss
    
    
    ret = laab_results.apply_cutoff(cutoff=cutoff)
    cutoff_results = ret.results
    score = ret.score
    
    prec=3
    
    inject = {
        "eb_name": laab_results.eb_version,
        "system": laab_results.system,
        "cpu_model": laab_results.cpu_model,
        "losses": format_floats_recursive(losses,prec),
        "mean_loss": f"{mean_loss:.{prec}f}",
        "cutoff_results": format_cutoff_results_md(cutoff_results),
        "score": score,
        "num_tests": len(losses),
        "times": format_floats_recursive(min_exec_times,prec),
        "cutoff": f"{cutoff:.2f}",
        "config": config,
        "exp_config": exp_config
    }
    
    with open(template_file, "r") as f:
        template_content = f.read()
        template = Template(template_content)
        report = template.render(**inject)
    
    with open(outfile, "w") as f:
        f.write(report)
    print(f"Report written to {outfile}")
    

    
     

