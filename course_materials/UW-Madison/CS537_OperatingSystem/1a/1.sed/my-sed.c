#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void removeWord(char* line, char* old_str)
{
    // remove end '\n'
    if (line[strlen(line) - 1] == '\n')
        line[strlen(line) - 1] = '\0';

    char* substr_bgn = strstr(line, old_str);
    if (substr_bgn == NULL)
        return;

    // printf("%zd, %zd", strlen(line), strlen(old_str));
    // printf("strcmp: %d", strcmp(line, old_str));

    // ("bb", "bb") -> line = "\0"
    if (strcmp(line, old_str) == 0) {
        strcpy(line, "\0");
        return;
    }

    char tail[1000];
    strcpy(tail, substr_bgn + strlen(old_str));
    // printf("tail: %s", tail);

    // ("aabb", "bb") -> line = "aa"
    if (strlen(tail) == 0) {
        strcpy(substr_bgn, "\0");
        return;
    }
    // ("bbcc", "bb") -> line = "cc"
    if (substr_bgn == line) {
        strcpy(line, tail);
        return;
    }
    // ("aabbcc", "bb") -> line = "aacc"
    strcpy(substr_bgn, "\0");
    strcpy(substr_bgn, tail);
}

void replaceWord(char* line, char* old_str, char* new_str)
{
    // remove end '\n'
    if (line[strlen(line) - 1] == '\n')
        line[strlen(line) - 1] = '\0';

    char* substr_bgn = strstr(line, old_str);
    if (substr_bgn == NULL){
        // printf("null");
        return;
    } 

    // printf("%s, %s\n", old_str, line);
    // printf("strcmp: %d", strcmp(line, old_str));

    // ("bb", "bb", "dd") -> line = "dd"
    if (strcmp(line, old_str) == 0) {
        strcpy(line, new_str);
        return;
    }

    char tail[1000];
    strcpy(tail, substr_bgn + strlen(old_str));
    // printf("tail: %s", tail);

    // ("aabb", "bb", "dd") -> line = "aadd"
    if (strlen(tail) == 0) {
        strcpy(substr_bgn, new_str);
        return;
    }
    // ("bbcc", "bb", "dd") -> line = "ddcc"
    if (substr_bgn == line) {
        strcpy(line, new_str);
        strcat(line, tail);
        return;
    }
    // ("aabbcc", "bb") -> line = "aaddcc"
    strcpy(substr_bgn, "\0");
    strcat(line, new_str);
    strcat(line, tail);
}

void myPrint(FILE* fp, char* old_str, char* new_str)
{
    char line[1000];
    char results[1000][1000];
    int count_result = 0;

    if (strcmp(new_str, """") == 0) {
        while (fgets(line, 1000, fp)) {
            removeWord(line, old_str);
            if (strlen(line) == 0)
                continue;
            strcpy(results[count_result], line);
            count_result++;       
        }
    } else {
        // printf("replace\n");
        while (fgets(line, 1000, fp)) {
            replaceWord(line, old_str, new_str);
            strcpy(results[count_result], line);
            count_result++;       
        }
    }
    
    for (int i = 0; i < count_result; i++)
        printf("%s\n", results[i]);
}

int main(int argc, char *argv[])
{
    if (argc <= 2) {
        printf("my-sed: find_term replace_term [file ...]\n");
        exit(1);
    }
    else {
        char old_str[1000];
        char new_str[1000];
        strcpy(old_str, argv[1]);
        strcpy(new_str, argv[2]);
        
        if (argc == 3){
            myPrint(stdin, old_str, new_str);
        } else {
            for (int i = 3; i < argc; i++) {
                FILE *fp = fopen(argv[i], "r");
                if (fp == NULL) {
                    printf("my-sed: cannot open file\n");
                    exit(1);
                }
                myPrint(fp, old_str, new_str);
                fclose(fp);
            }
        }
    }
    exit(0);
}