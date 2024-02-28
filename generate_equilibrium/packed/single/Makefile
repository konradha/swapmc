CC=clang++-17
CFLAGS =-I. -pedantic -ffast-math -march=native -O3 -Wall -fopenmp -Wunknown-pragmas  -lm -lstdc++ -std=c++17
TARGET = to_sim
SRC = sim.cpp
RUN_SCRIPT = python run_local.py 


$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) 

run: $(TARGET)
	$(RUN_SCRIPT)

clean:
	rm $(TARGET) 

