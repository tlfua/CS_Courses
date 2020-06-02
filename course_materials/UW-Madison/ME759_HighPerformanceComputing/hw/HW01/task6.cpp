#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);

    for (int i = 0; i <= N; i++) {
        std::cout << i;
        if (i < N) {
            std::cout << " ";
        } else {
            std::cout << "\n";
        }
    }

    return 0;
}