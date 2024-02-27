CC=mpicxx
CFLAGS =-I. -ftree-vectorize -pedantic -ffast-math -march=native -O3 -Wall -fopenmp -Wunknown-pragmas  -lm -lstdc++ -std=c++17
TARGET = to_hybrid
SRC = hybrid-single-beta-no-pt.cpp
RUN_SCRIPT = runfile.sh 


$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) 

run: $(TARGET)
	bash runfile.sh

clean:
	rm $(TARGET) 

