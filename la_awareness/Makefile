.PHONY: pyt scr

export LAAB_REPS = 3
export LAAB_N = 3000
CORE ?= 0

pyt_cpu:
	mkdir -p $(LAAB_PYT_CPU_RESULTS_DIR)
	rm -f $(LAAB_PYT_CPU_RESULTS_DIR)/data.txt
	@LAAB_PREFIX='taskset -c $(CORE)' \
		$(MAKE) --no-print-directory -C src/PyTorch/v2-cpu | tee -a $(LAAB_PYT_CPU_RESULTS_DIR)/data.txt
	@echo "Results saved to $(LAAB_PYT_CPU_RESULTS_DIR)/data.txt"
