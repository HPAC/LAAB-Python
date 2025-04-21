oblas:
	gcc -O2  sgemm.c -c
	gcc sgemm.o -o sgemm -lopenblas

fblas:
	gcc -O2  sgemm.c -c
	gcc sgemm.o -o sgemm -lflexiblas

run:
	taskset -c 0 ./sgemm 3000 3000 3000

clean: 
	rm sgemm.o sgemm

