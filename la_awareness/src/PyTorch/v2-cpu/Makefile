
PREFIX = taskset -c 0
REPS = 3
N = 3000

SCRIPTS = \
	cse_addition.py \
	cse_matmul_paranthesis.py \
	cse_matmul_no_paranthesis.py \
	matchain_ltor.py \
	matchain_rtol.py \
	matchain_mixed.py \
	mp_trmm.py \
	mp_syrk.py \
	mp_tridiag.py \
	mp_diag.py \
	am_distributivity1.py \
	am_distributivity2.py \
	am_blocked.py \
	cm_loops.py \
	cm_partial_op_sum.py \
	cm_partial_op_prod.py \

run: $(SCRIPTS:.py=.run)

# Pattern rule: how to run each script
%.run: %.py
	LAAB_REPS=$(REPS) LAAB_N=$(N) $(PREFIX) python $<
	