all:
	gcc -O2 sgemm.c -c

obtrace:
	gcc -fPIC -shared -o obtrace.so obtrace.c -ldl

run_mkl:
	gcc sgemm.o -o sgemm -lmkl_rt
	MKL_NUM_THREADS=1 LD_PRELOAD=./obtrace.so ./sgemm 3000 3000 3000

run_oblas:
	gcc sgemm.o -o sgemm -lopenblas
	LD_PRELOAD=./obtrace.so ./sgemm 3000 3000 3000

run_fblas:
	gcc sgemm.o -o sgemm -lflexiblas
	LD_PRELOAD=./obtrace.so ./sgemm 3000 3000 3000

run_scipy:
	LD_PRELOAD=./obtrace.so python gemm_scipy.py

run_torch:
	LD_PRELOAD=./obtrace.so python gemm_torch.py

run_tf:
	LD_PRELOAD=./obtrace.so python gemm_tf.py

run_np:
	LD_PRELOAD=./obtrace.so python gemm_np.py

run_nn:
	LD_PRELOAD=./obtrace.so python nn_torch.py

run_gls:
	LD_PRELOAD=./obtrace.so python gls.py

run_gpt:
	LD_PRELOAD=./obtrace.so python gpt.py

run_dlsym:
	gcc sgemm_dl.c -o sgemm_dl -ldl
	LD_PRELOAD=./obtrace.so ./sgemm_dl


clean:
	rm -f *.o sgemm sgemm_dl
	rm obtrace.so
