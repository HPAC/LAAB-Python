import os
import numpy as np
from types import SimpleNamespace

class LAABResults:
    def __init__(self, data_file, eb_version, system, delta=0.002):
        self.eb_version = eb_version
        self.system = system
        self.cpu_model =  ""
        self.data = {}
        self.delta = delta
        
        self._prepare(data_file)
        
        self.slow_down = {}
        self.loss = {}
        self.mean_loss = 0.0
        
    def _prepare(self, data_file):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found")
        
        with open(data_file, "r") as f:
            for line in f:
                self._add_entry(line)
                
        ## sanity check
        for exp, data in self.data.items():
            if len(data['ref_positive']) == 0:
                raise ValueError(f"No ref_positive data for experiment {exp}")
            if len(data['ref_negative']) == 0:
                raise ValueError(f"No ref_negative data for experiment {exp}")
            for test, values in data['tests'].items():
                if len(values) == 0:
                    raise ValueError(f"No data for test {test} in experiment {exp}")
                    
    
    def _add_entry(self, line):
   
        if line.startswith("Model name:"):
            self.cpu_model = line.split("Model name:")[1].strip()
        
        if not line.startswith("[LAAB]"):
            return
              
        record = line.split("[LAAB]")[1].strip().split("|")[1:]
    
        exp = record[0].strip()
        if not exp in self.data:
            self.data[exp] = {'ref_positive': [],  'tests':{}, 'ref_negative': []}
        
        for i in range(1, len(record)):
            test = record[i].strip().split("=")[0].strip()
            value = record[i].split('=')[1].split(' ')[0].strip()
            
            if test == "ref_positive":
                value = max(float(value), self.delta)
                self.data[exp]['ref_positive'].append(value)
            elif test == "ref_negative":
                if not value.startswith("R"):
                    value = float(value)
                    self.data[exp]['ref_negative'].append(value)
                else:
                    if len(self.data[exp]['ref_negative']) > 0:
                        continue
                    if value[1] == '+':
                        self.data[exp]['ref_negative'] = self.data[value[2:]]['ref_positive']
                    elif value[1] == '-':
                        self.data[exp]['ref_negative'] = self.data[value[2:]]['ref_negative']
            else:
                if not test in self.data[exp]['tests']:
                    self.data[exp]['tests'][test] = []
                self.data[exp]['tests'][test].append(float(value))
                
    def compute_loss(self, q_min=25, q_max=75):
        if not self.data:
            raise ValueError("No data to compute results")
        _loss_vals = []
        for exp, data in self.data.items():
            self.slow_down[exp] = {}
            self.slow_down[exp]['ref_negative'] = self._slow_down(data['ref_negative'], data['ref_positive'], q_min=q_min, q_max=q_max)
            self.loss[exp] = {}
            for test, values in data['tests'].items():
                self.slow_down[exp][test] = self._slow_down(values, data['ref_positive'], q_min=q_min, q_max=q_max)
                
                self.loss[exp][test] = 0.0
                if self.slow_down[exp]['ref_negative'] > 0.0:
                    self.loss[exp][test] = self.slow_down[exp][test]/self.slow_down[exp]['ref_negative']
            
            min_loss = np.min(list(self.loss[exp].values()))
            self.loss[exp]["result"] = min_loss
            _loss_vals.append(min_loss)
            
        self.mean_loss = np.mean(_loss_vals)
                
            
    
    def _slow_down(self, vals, ref_vals, q_min=25, q_max=75):
        vals_q = np.percentile(vals, [q_min, q_max])
        # ref_vals = [max(x, delta) for x in ref_vals]
        ref_vals_q = np.percentile(ref_vals, [q_min, q_max])
        
        slow_down = 0.0
        if vals_q[0] > ref_vals_q[1]:
            # loss is diff in median
            slow_down = (np.min(vals) - np.min(ref_vals)) / np.min(ref_vals)
        
        return slow_down
    
    def get_min_test_times(self):
        min_time = {}
        for exp, data in self.data.items():
            min_time[exp] = {'ref_positive': 0,  'tests':{}, 'ref_negative':0}
            min_time[exp]['ref_positive'] = np.min(data['ref_positive'])
            min_time[exp]['ref_negative'] = np.min(data['ref_negative'])
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