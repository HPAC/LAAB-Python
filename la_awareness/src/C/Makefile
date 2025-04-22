# LAAB_PREFIX = taskset -c 0
# LAAB_REPS = 3
# LAAB_N = 3000
# LD_BLAS = -lopenblas

run:
	gcc -O2  sgemm.c -c
	gcc sgemm.o -o sgemm $(LD_BLAS)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) ./sgemm $(LAAB_N) $(LAAB_N) $(LAAB_N)

# oblas:
# 	gcc -O2  sgemm.c -c
# 	gcc sgemm.o -o sgemm -lopenblas

# fblas:
# 	gcc -O2  sgemm.c -c
# 	gcc sgemm.o -o sgemm -lflexiblas

# run:
# 	taskset -c 0 ./sgemm 3000 3000 3000

clean: 
	rm sgemm.o sgemm

