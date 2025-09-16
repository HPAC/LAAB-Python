import os
import numpy as np
from types import SimpleNamespace

class LAABResults:
    def __init__(self, data_file, eb_version, system):
        self.eb_version = eb_version
        self.system = system
        self.cpu_model =  ""
        self.data = {}
        
        self._prepare(data_file)
        
        self.loss = {}
        self.mean_loss = 0.0
        
    def _prepare(self, data_file):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found")
        
        with open(data_file, "r") as f:
            for line in f:
                self._add_entry(line)
    
    def _add_entry(self, line):
   
        if line.startswith("Model name:"):
            self.cpu_model = line.split("Model name:")[1].strip()
        
        if not line.startswith("[LAAB]"):
            return
              
        record = line.split("[LAAB]")[1].strip().split("|")[1:]
    
        exp = record[0].strip()
        if not exp in self.data:
            self.data[exp] = {'optimized': [],  'tests':{}}
        
        for i in range(1, len(record)):
            test = record[i].strip().split("=")[0].strip()
            value = float(record[i].split('=')[1].split(' ')[0].strip())
            
            if test == "optimized":
                self.data[exp]['optimized'].append(value)
            else:
                if not test in self.data[exp]['tests']:
                    self.data[exp]['tests'][test] = []
                self.data[exp]['tests'][test].append(value)
                
    def compute_loss(self, q_min=25, q_max=75):
        if not self.data:
            raise ValueError("No data to compute results")
        _loss_vals = []
        for exp, data in self.data.items():
            self.loss[exp] = {}
            for test, values in data['tests'].items():
                loss = self._loss_fn(values, data['optimized'],q_min=q_min, q_max=q_max)
                self.loss[exp][test] = loss
            _loss_vals.append(np.min(list(self.loss[exp].values())))
        self.mean_loss = np.mean(_loss_vals)
                
            
    
    def _loss_fn(self, vals, ref_vals, q_min=25, q_max=75):
        vals_q = np.percentile(vals, [q_min, q_max])
        ref_vals_q = np.percentile(ref_vals, [q_min, q_max])
        
        loss = 0.0
        if vals_q[0] > ref_vals_q[1]:
            # loss is diff in median
            loss = (np.min(vals) - np.min(ref_vals)) / np.min(ref_vals)
        
        return loss
    
    def get_min_test_times(self):
        min_time = {}
        for exp, data in self.data.items():
            min_time[exp] = {'optimized': 0,  'tests':{}}
            min_time[exp]['optimized'] = np.min(data['optimized'])
            for test, values in data['tests'].items():
                min_time[exp]['tests'][test] = np.min(values)
        return min_time
    
    def apply_cutoff(self, cutoff):
        if not self.loss:
            raise ValueError("No loss computed. Call compute_loss() first.")
        
                
        exp_results = {}
        score = 0
        for exp, tests in self.loss.items():
            exp_results[exp] = {}
            for test, loss in tests.items():
                exp_results[exp][test] = loss <= cutoff
            if True in exp_results[exp].values():
                score += 1
        
        results = {
            'score': score,
            'results': exp_results
        }
        
        return SimpleNamespace(**results)