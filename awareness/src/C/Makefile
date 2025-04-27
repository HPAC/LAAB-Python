# LAAB_PREFIX = OMP_NUM_THREADS=58
# LAAB_REPS = 5
# LAAB_N = 3000
# LD_BLAS = -lmkl_rt


sgemm:
	gcc -O3  sgemm.c -c
	gcc sgemm.o -o sgemm $(LD_BLAS)

ssyrk:
	gcc -O3  ssyrk.c -c
	gcc ssyrk.o -o ssyrk $(LD_BLAS)

strmm:
	gcc -O3  strmm.c -c
	gcc strmm.o -o strmm $(LD_BLAS)

stridiagmm:
	gcc -O3  stridiagmm.c -c -fopenmp
	gcc stridiagmm.o -o stridiagmm -fopenmp

run: sgemm ssyrk strmm stridiagmm
	REP=$(LAAB_REPS) $(LAAB_PREFIX) ./sgemm $(LAAB_N) $(LAAB_N) $(LAAB_N)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) ./ssyrk $(LAAB_N) $(LAAB_N)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) ./strmm $(LAAB_N) $(LAAB_N)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) ./stridiagmm $(LAAB_N) $(LAAB_N)
	rm *.o sgemm ssyrk strmm stridiagmm

clean: 
	rm sgemm.o sgemm ssyrk.o ssyrk strmm.o strmm stridiagmm.o stridiagmm

