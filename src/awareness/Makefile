.PHONY: scr

#CORE ?= 0
export LAAB_REPS = 5
export LAAB_N = 3000
#export LAAB_PREFIX=taskset -c $(CORE)

pyt_v2_cpu:
	mkdir -p ../../$(LAAB_PYT_CPU_RESULTS_DIR)
	rm -f ../../$(LAAB_PYT_CPU_RESULTS_DIR)/data.txt
	lscpu | grep "Model name" > ../../$(LAAB_PYT_CPU_RESULTS_DIR)/data.txt
	@$(MAKE) --no-print-directory -C src/PyTorch/v2-cpu | tee -a ../../$(LAAB_PYT_CPU_RESULTS_DIR)/data.txt
	@echo "Results saved to ../../$(LAAB_PYT_CPU_RESULTS_DIR)/data.txt"
	python prepare_report.py src/PyTorch/v2-cpu/report.md ../../$(LAAB_PYT_CPU_RESULTS_DIR)/data.txt

tf_v2_cpu:
	mkdir -p ../../$(LAAB_TF_CPU_RESULTS_DIR)
	rm -f ../../$(LAAB_TF_CPU_RESULTS_DIR)/data.txt
	lscpu | grep "Model name" > ../../$(LAAB_TF_CPU_RESULTS_DIR)/data.txt
	@$(MAKE) --no-print-directory -C src/TensorFlow/v2-cpu | tee -a ../../$(LAAB_TF_CPU_RESULTS_DIR)/data.txt
	@echo "Results saved to ../../$(LAAB_TF_CPU_RESULTS_DIR)/data.txt"
	python prepare_report.py src/TensorFlow/v2-cpu/report.md ../../$(LAAB_TF_CPU_RESULTS_DIR)/data.txt

jax_v0_cpu:
	mkdir -p ../../$(LAAB_JAX_CPU_RESULTS_DIR)
	rm -f ../../$(LAAB_JAX_CPU_RESULTS_DIR)/data.txt
	lscpu | grep "Model name" > ../../$(LAAB_JAX_CPU_RESULTS_DIR)/data.txt
	@$(MAKE) --no-print-directory -C src/jax/v0-cpu | tee -a ../../$(LAAB_JAX_CPU_RESULTS_DIR)/data.txt
	@echo "Results saved to ../../$(LAAB_JAX_CPU_RESULTS_DIR)/data.txt"
	python prepare_report.py src/jax/v0-cpu/report.md ../../$(LAAB_JAX_CPU_RESULTS_DIR)/data.txt

all: pyt_v2_cpu tf_v2_cpu jax_v0_cpu
