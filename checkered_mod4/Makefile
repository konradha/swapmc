CC=clang++-17
CFLAGS =-ftree-vectorize -pedantic -march=native -O3 -Wall -Wunknown-pragmas -fopenmp  -lm -lstdc++ -std=c++20
TARGET = checkerboard
SRC = sweep.cpp 
RUN_SCRIPT = runfile.sh 

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) 

run: $(TARGET)
	bash ${RUN_SCRIPT}

testclean:
	rm run_L*.txt

clean:
	rm $(TARGET)
