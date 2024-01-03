#include <mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    for (int i=0;i<10;++i)
        std::cout << MPI_Wtime() << " ";
    std::cout << "\n";
    for (int i=0;i<10;++i)
        std::cout << (int)((1<<23) * MPI_Wtime()) << " ";
    std::cout << "\n";
}
