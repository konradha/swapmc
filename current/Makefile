CC=clang-15
CFLAGS = -pedantic -Wall -Wunknown-pragmas -fsanitize=address -lm -lstdc++ -std=c++17 -ffast-math -O3 -march=native
TARGET = lat
SRC = driver.cpp 
DEPS = lattice.h

$(TARGET): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run:
	bash $(CURDIR)/run.sh 

clean:
	rm -f $(TARGET)
