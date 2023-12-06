CC=clang-15
CFLAGS = -fsanitize=address -lm -lstdc++ -std=c++17 -ffast-math -O3 -march=native
TARGET = lat
SRC = lattice3d.cpp
DEPS = structs.h

$(TARGET): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run:
	bash $(CURDIR)/run.sh 

clean:
	rm -f $(TARGET)
