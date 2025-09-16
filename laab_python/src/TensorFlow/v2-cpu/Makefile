.PHONY: run

# Uncomment the following  if you want to run this makefile independently
# LAAB_PREFIX = taskset -c 0
# LAAB_REPS = 3
# LAAB_N = 3000


SCRIPTS = \
	sgemm.py \
	cse_addition.py \
	cse_matmul_paranthesis.py \
	cse_matmul_no_paranthesis.py \
	cse_suboptimal.py \
	matchain_ltor.py \
	matchain_rtol.py \
	matchain_mixed.py \
	mp_trmm.py \
	mp_syrk.py \
	mp_tridiag.py \
	am_distributivity1.py \
	am_distributivity2.py \
	am_blocked.py \
	cm_loops.py \
	cm_partial_op_sum.py \
	cm_partial_op_prod.py \

all: run_c run_py

run_c:
	LAAB_REPS=$(LAAB_REPS) LAAB_N=$(LAAB_N) LAAB_PREFIX='$(LAAB_PREFIX)' $(MAKE) -C ../../C/ run

run_py: $(SCRIPTS:.py=.run)

# Pattern rule: how to run each script
%.run: %.py
	LAAB_REPS=$(LAAB_REPS) LAAB_N=$(LAAB_N) $(LAAB_PREFIX) python $< 
	