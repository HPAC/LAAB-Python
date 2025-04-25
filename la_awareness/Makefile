.PHONY: pyt scr

CORE ?= 0
export LAAB_REPS = 3
export LAAB_N = 3000
#export LAAB_PREFIX=taskset -c $(CORE)

pyt_cpu:
	mkdir -p $(LAAB_PYT_CPU_RESULTS_DIR)
	rm -f $(LAAB_PYT_CPU_RESULTS_DIR)/data.txt
	@$(MAKE) --no-print-directory -C src/PyTorch/v2-cpu | tee -a $(LAAB_PYT_CPU_RESULTS_DIR)/data.txt
	@echo "Results saved to $(LAAB_PYT_CPU_RESULTS_DIR)/data.txt"
