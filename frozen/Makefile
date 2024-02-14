CC=mpicxx
CFLAGS =-ftree-vectorize -pedantic -ffast-math -march=native -O3 -Wall -fopenmp -Wunknown-pragmas  -lm -lstdc++ -std=c++17
TARGET = to_single
SRC = single_tempering.cpp 
RUN_SCRIPT = runfile.sh 

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) 

4: $(TARGET)
	bash runfile_4.sh

8: $(TARGET)
	bash runfile_8.sh

clean:
	rm $(TARGET) 

