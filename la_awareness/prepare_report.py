import os
import sys
import statistics
from jinja2 import Template


#def evaluate(value,ref_value, exp_name):
#    pass    

def compare(value, ref_value):
    mean_value = statistics.mean(value[1:])
    mean_ref_value = statistics.mean(ref_value[1:])
    if mean_value > mean_ref_value*1.40:
        return ":x:",0
    return ":white_check_mark:",1

if __name__ == "__main__":
    
    error_msg = f"Usage: {os.path.basename(__file__)} <path_to_template> <path_to_data>\n"
    
    if len(sys.argv) != 3:
        print(error_msg)
        sys.exit(1)

    ## Get the template file
    template_file = sys.argv[1]
    if not os.path.exists(template_file):
        print(f"Path {template_file} does not exist.")
        sys.exit(1)    
    print(f"Using template: {template_file}")
    
    ## Get the data file
    data_file = sys.argv[2]
    if not os.path.exists(data_file):
        print(f"Path {data_file} does not exist.")
        sys.exit(1)
    print(f"Using data: {data_file}")
    
    ## Identify the EB version and system from the data file path
    _parts = data_file.split(os.sep)
    eb_name = f"{_parts[1]}/{_parts[2]}"
    system = _parts[-2]    
    print(f"Preparing report for {eb_name} on {system}")
    
    inject = {
        "eb_name": eb_name,
        "system": system,
    }
    
 
    # Read the data file
    data = []
    cpu_model = ""
    with open(data_file, "r") as f:
        for line in f:
            if line.startswith("[LAAB]"):
                data.append(line.split("[LAAB]")[1].strip().split("|")[1:])
            if line.startswith("Model name:"):
                cpu_model = line.split("Model name:")[1].strip()
                inject["cpu_model"] = cpu_model

    ## Example data:
        # [' sgemm ', ' optimized=0.462 s']
        # [' sgemm ', ' optimized=0.461 s']
        # [' sgemm ', ' optimized=0.461 s']
        # [' sgemm ', ' actual=0.48239 s']
        # [' sgemm ', ' actual=0.47505 s']
        # [' sgemm ', ' actual=0.47254 s']
        # [' cse_addition ', ' optimized=0.48919 s ', ' actual=0.49557 s']
        # [' cse_addition ', ' optimized=0.48985 s ', ' actual=0.49156 s']
        # [' cse_addition ', ' optimized=0.48931 s ', ' actual=0.49143 s']

    exp_set = set()
    ## Loop through the data and prepare a dict for injecting into the template
    for x in data:
        exp = x[0].strip()
        exp_set.add(exp)
        for i in range(1, len(x)):
            key = f"{exp}_{x[i].split('=')[0].strip()}"
            value = float(x[i].split('=')[1].split(' ')[0].strip())
            
            if not key in inject:
                inject[key] = []
            inject[key].append(value)
    
    ## Example inject dict:
        # sgemm_optimized: [0.462, 0.461, 0.461]
        # sgemm_actual: [0.48239, 0.47505, 0.47254]
        # cse_addition_optimized: [0.48919, 0.48985, 0.48931]
        # cse_addition_actual: [0.49557, 0.49156, 0.49143]  
    
    # Inject results by comparing the optimized values with other keys
    score = 0
    for exp in exp_set:
        passed = False
        ref = f"{exp}_optimized"
        matching_keys = [k for k in inject.keys() if k.startswith(exp) and k != ref]
        #print(f"Matching keys for {exp}: {matching_keys}, ref_key: {ref}")      
        
        for key in matching_keys:
            value = inject[key]
            ref_value = inject[ref]
            
            result = compare(value, ref_value)
            if result[1] == 1:
                passed = True
            
            
            print(f"Result for {key}: {result[1]}")
            inject[f'{key}_result'] = result[0]
     
        if passed:
            score += 1
            
    inject["score"] = score
    inject["total"] = len(exp_set)
    print(f"Score: {score}/{len(exp_set)}")   
    
    # Convert the list of values to the mean value excluding the first value
    for key,value in inject.items():
        if key in ["eb_name", "system"]:
            continue
        
        #check if val is instance of list
        if isinstance(value, list):
            value = statistics.mean(value[1:])
            inject[key] = "{:.4f}".format(value)
            
        #print(f"{key}: {inject[key]}")
        
  
    ## Prepare the report
    with open(template_file, "r") as template:
        template_content = template.read()
    template = Template(template_content)
    rendered_template = template.render(inject)
  
    report_file = os.path.join(os.path.dirname(data_file), "README.md")
    with open(report_file, "w") as f:
        f.write(rendered_template)
    print(f"Report written to {report_file}")
    