
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void myPrint(FILE * fp)
{
    char prev_line[6000];
    char cur_line[6000];
    int count_line = 0;
    char results[1000][6000];
    int count_result = 0;
    while (fgets(cur_line, 6000, fp)) {
        // remove end '\n'
        if (cur_line[strlen(cur_line) - 1] == '\n')
            cur_line[strlen(cur_line) - 1] = '\0';

        if (count_line == 0) {
            // printf("%s", cur_line);
            strcpy(results[count_result], cur_line);
            count_result++;
                
            strcpy(prev_line, cur_line);
            count_line++;
            continue;
        }

        if (strcmp(cur_line, prev_line) != 0) {
            // printf("%s", cur_line);
            strcpy(results[count_result], cur_line);
            count_result++;
        }

        strcpy(prev_line, cur_line);
        count_line++;
    }

    for (int i = 0; i < count_result; i++)
        printf("%s\n", results[i]);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {       // do not specify files
        myPrint(stdin);
    } else {
        for (int i = 1; i < argc; i++) {
            FILE *fp = fopen(argv[i], "r");
            if (fp == NULL) {
                printf("my-uniq: cannot open file\n");
                exit(1);
            }
            myPrint(fp);
            fclose(fp);
        }
    }

    // printf("\n");
    exit(0);
}