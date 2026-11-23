OMP_xl : 
	xlC_r -O1 -qstrict -qarch=pwr8 -qtune=pwr8 -qsmp=omp -I ./include -std=c++11 ./OMP.cpp ./source/solver.cpp -o OMP

xl : 
	xlC_r -O1 -qstrict -qarch=pwr8 -qtune=pwr8 -I ./include -std=c++11 ./OMP.cpp ./source/solver.cpp -o SEQ

OMP_xl_dump_errs : 
	xlC_r -O1 -qstrict -qarch=pwr8 -qtune=pwr8 -qsmp=omp -DDUMP_ERRS -I ./include -std=c++11 ./OMP.cpp ./source/solver.cpp -o OMP

OMP_xl_dump_diff : 
	xlC_r -O1 -qstrict -qarch=pwr8 -qtune=pwr8 -qsmp=omp -DDUMP_DIFF_EVERY=300 -I ./include -std=c++11 ./OMP.cpp ./source/solver.cpp -o OMP

MPI_xl : 
	mpixlC -O1 -qstrict -qarch=pwr8 -qtune=pwr8 -I ./include -std=c++11 ./MPI.cpp ./source/solver.cpp -o MPI