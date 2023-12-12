CC=clang-15
CFLAGS = -pedantic -Wall -Wunknown-pragmas -fsanitize=address -lm -lstdc++ -std=c++17 -ffast-math -O3 -march=native
CFLAGS_OMP = -fopenmp -lgomp -pedantic -Wall -Wunknown-pragmas -fsanitize=address -lm -lstdc++ -std=c++17 -ffast-math -O3 -march=native
#CFLAGS = -lm -lstdc++ -std=c++17 -ffast-math -O3 -march=native
TARGET = lat
TARGET_PAR = lat_par
SRC = lattice3d.cpp
SRC_PAR = omp_lattice3d.cpp 
DEPS = structs.h

$(TARGET): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

parallel: $(SRC_PAR) $(DEPS)
	$(CC) $(CFLAGS_OMP) $(SRC_PAR) -o $(TARGET_PAR)

run:
	bash $(CURDIR)/run.sh 

clean:
	rm -f $(TARGET)
