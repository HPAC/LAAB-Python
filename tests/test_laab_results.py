from laab_python.laab_results import LAABResults

def test1():
    
    data_file = "tests/sample_data/data_pyt.txt"
    # data_file = "tests/sample_data/data_tf.txt"
    eb_version = "PyTorch/2.1.2-foss-2023a"
    system = "HPC2N_x86_64"

    laab_results = LAABResults(data_file, eb_version, system)
    
    print("METADATA")
    print(laab_results.cpu_model)
    print(laab_results.eb_version)
    print(laab_results.system)
    
    print("\nDATA")
    for exp, data in laab_results.data.items():
        print(f"{exp}: {data}")

    
    min_test_times = laab_results.get_min_test_times()
    print("\nMIN TEST TIMES")
    for exp, data in min_test_times.items():
        print(f"{exp}: {data}")
    
    laab_results.compute_loss()
    
    print("\nSLOW_DOWN")
    for exp, slow_down in laab_results.slow_down.items():
        print(f"{exp}: {slow_down}")
    
    print("\nLOSS")
    for exp, loss in laab_results.loss.items():
        print(f"{exp}: {loss}")
    

    print(f"Mean loss: {laab_results.mean_loss}")
    
    results = laab_results.apply_cutoff(cutoff=0.05)    
    print("\nCUTOFF RESULTS")    
    
    for exp, result in results.results.items():
        print(f"{exp}: {result}")
    
    
    print(f"Overall score: {results.score}/{len(laab_results.loss)}")
    
if __name__ == "__main__":
    test1()