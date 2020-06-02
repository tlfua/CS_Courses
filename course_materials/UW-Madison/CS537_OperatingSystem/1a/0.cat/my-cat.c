
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc < 2)
        exit(0);

    for (int i = 1; i < argc; i++) {
        FILE *fp = fopen(argv[i], "r");
        if (fp == NULL) {
            printf("my-cat: cannot open file\n");
            exit(1);
        }

        char line[1000];
        while (fgets(line, 1000, fp)) {
            printf("%s", line);
        }
        fclose(fp);
    }
    // printf("\n");

    exit(0);
}