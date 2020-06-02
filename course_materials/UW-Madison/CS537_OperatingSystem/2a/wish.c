#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>

#define MAX_HISTORY_ROW_COUNT 100
#define MAX_PATH_COUNT 100

// Operation Code
#define ERROR -1
#define NORMAL 0
#define REDIR 1
#define PIPE 2

char **history;
int history_row_count = 0;

char default_path[5] = "/bin/";
char **paths;
int path_count = 0;

char error_message[30] = "An error has occurred\n";

void call_exit(int* tokens_size)
{
    if (*tokens_size != 1){
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message));
        return;
    }   
    // free history
    for (int i=0; i<history_row_count; i++)
        free(history[i]);
    // free paths
    for (int i=0; i<path_count; i++)
        free(paths[i]);
    
    exit(0);
}

void call_cd(char *tokens[50], int* tokens_size)
{
    if (*tokens_size != 2){
        // printf("call_cd: A\n");
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message));
        return;
    }
    
    int chdir_ret = chdir(tokens[1]);
    if(chdir_ret != 0){
        // printf("call_cd: B\n");
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message));
        return;
    }
}

void call_history(char *tokens[50], int* tokens_size)
{
    if (*tokens_size > 2){
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message));
        return;
    }
    int bgn;
    int tail_size = 0;
    if (*tokens_size == 1){
        bgn = 0;
    } else {
        // tail_size = atoi(tokens[1]);
        tail_size = (int)ceil(atof(tokens[1]));
        if (tail_size < 1)
            return;
        if (tail_size >= history_row_count){
            bgn = 0;
        } else {
            bgn = history_row_count - tail_size;
        }
    }
    for (int i=bgn; i<history_row_count; i++)
        printf("%s", history[i]);
}

void call_path(char *tokens[50], int* tokens_size)
{
    // FREE
    for (int i=0; i<path_count; i++){
        free(paths[i]);
    }
    path_count = 0;

    if (*tokens_size == 1)
        return;
    for (int token_index=1; token_index<*tokens_size; token_index++){
        paths[path_count] = (char*)malloc(strlen(tokens[token_index]) * sizeof(char));
        strcpy(paths[path_count], tokens[token_index]);
        if (paths[path_count][strlen(paths[path_count])-1] != '/')
            strncat(paths[path_count], "/", 2); // why not 1 ??
        path_count++;
    }
    // for (int i=0; i<path_count; i++)
    //     printf("%s\n", paths[i]);
}

// void run_executable(char* first_token)
void run_executable(char *tokens[50], int* tokens_size, char* redir_path)
// void run_executable(char **tokens, int* tokens_size)
{
    char* first_token = tokens[0];

    int fork_count = 1;
    int fork_ret = fork();
    int child_completed = -1;
    if (fork_ret < 0){
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message)); 
    } else if (fork_ret > 0){ // parent
        fork_count++;
    } else { // child
        if (redir_path != NULL){
            freopen(redir_path, "w+", stdout);
        }
        char *cur_path;
        for (int i=0; i<path_count; i++){
            cur_path = malloc(sizeof(char) * strlen(paths[i]));
            strcpy(cur_path, paths[i]);
            strncat(cur_path, first_token, strlen(first_token));
            if(access(cur_path, X_OK) == 0){
                execv(cur_path, tokens);
                child_completed = 0;
                break;
            }
            free(cur_path);
        }
        if(child_completed != 0){
            char error_message[30] = "An error has occurred\n";
            write(STDERR_FILENO, error_message, strlen(error_message));
        }
        exit(0);
    }

    if(fork_ret > 0){ //parent wait for all child processes
        for(int q = fork_count; q > 0; q--){
            wait(NULL);
        }
    }
}

void run_piped_executables(char *first_tokens[50], char *second_tokens[50])
{
    int READ_END = 0;
    int WRITE_END = 1;

    int fd[2];
    pipe(fd);

    int rc1 = 0;
    int rc2 = 0;
    if ((rc1 = fork()) == 0) {
        dup2(fd[1], 1);
        close(fd[0]);

        char *cur_path;
        for (int i=0; i<path_count; i++){
            cur_path = malloc(sizeof(char) * strlen(paths[i]));
            strcpy(cur_path, paths[i]);
            strncat(cur_path, first_tokens[0], strlen(first_tokens[0]));
            if(access(cur_path, X_OK) == 0){
                execv(cur_path, first_tokens);
                break;
            }
            free(cur_path);
        }
        write(STDERR_FILENO, error_message, strlen(error_message));
        // printf ("in rc1\n");
    } else {
        if ((rc2 = fork()) == 0) {
            int status;
            waitpid(rc1, &status, WUNTRACED);

            dup2(fd[0], 0);
            close(fd[1]);

            char *cur_path;
            for (int i=0; i<path_count; i++){
                cur_path = malloc(sizeof(char) * strlen(paths[i]));
                strcpy(cur_path, paths[i]);
                strncat(cur_path, second_tokens[0], strlen(second_tokens[0]));
                if(access(cur_path, X_OK) == 0){
                    execv(cur_path, second_tokens);
                    break;
                }
                free(cur_path);
            }
            write(STDERR_FILENO, error_message, strlen(error_message));
            // printf ("in rc2\n");
        } else {
            int status;
            close(fd[READ_END]);
            close(fd[WRITE_END]);
            waitpid(rc2, &status, WUNTRACED);
        }
    }
}

void executePipedCommands(char *first_tokens[50], char *second_tokens[50])
{
    int rc = fork();
    if (rc < 0) { // fork failed
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message));
        exit(1);
    } else if (rc == 0) { // child

        run_piped_executables(first_tokens, second_tokens);
                
        exit(0); // leave child process
    } else { // parent
        wait(NULL);
    }
}

void executeCommand(char *tokens[50], int* tokens_size, char* redir_path)
{
    // ERROR for the a built-in command plus redirection
    if ( ( (strcmp(tokens[0], "exit")==0)||(strcmp(tokens[0], "cd")==0)||(strcmp(tokens[0], "history")==0)||(strcmp(tokens[0], "path")==0)) &&\
        (redir_path != NULL)) {
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message));
        return;
    }

    if (strcmp(tokens[0], "exit") == 0){
        call_exit(tokens_size);
    } else if (strcmp(tokens[0], "cd") == 0){
        call_cd(tokens, tokens_size);
    } else if (strcmp(tokens[0], "history") == 0){
        call_history(tokens, tokens_size);
    } else if (strcmp(tokens[0], "path") == 0){
        call_path(tokens, tokens_size);
    } else {
        run_executable(tokens, tokens_size, redir_path);
    }
}

void getOperationCode(char* raw_command, int* op_code)
{
    int count_redir = 0, count_pipe = 0;
    for (int i=0; i<strlen(raw_command); i++){
        if (raw_command[i] == '>')
            count_redir++;
        else if (raw_command[i] == '|')
            count_pipe++;
        if ((count_redir == 2) || (count_pipe == 2)){
            *op_code = ERROR;
            return;
        }
    }
    if ((count_redir == 1) && (count_pipe == 1)){
        *op_code = ERROR;
    } else if (count_redir == 1){
        *op_code = REDIR;
    } else if (count_pipe == 1){
        *op_code = PIPE;
    } else {
        *op_code = NORMAL;
    }
}

void splitIntoTwoCommands(char* raw_command, char* pair_command[2], int* pair_command_size)
{
    char *token;
    char *the_rest = raw_command;
    *pair_command_size = 0;
    while ((token = strtok_r(the_rest, ">|\n", &the_rest))) {
        // printf ("%s\n", token);
        pair_command[*pair_command_size] = (char*)malloc(strlen(token) * sizeof(char));
        strcpy(pair_command[*pair_command_size], token);
        (*pair_command_size)++;
    }
}

void parseRawCommand(char* raw_command, char *tokens[50], int* tokens_size)
{
    char *token;
    char *the_rest = raw_command;
    *tokens_size = 0;
    while ((token = strtok_r(the_rest, " \t\n", &the_rest))) {
        // printf ("%s\n", token);
        // if (strcmp(token, "\n") == 0)
        //     continue;
        tokens[*tokens_size] = (char*)malloc(strlen(token) * sizeof(char));
        strcpy(tokens[*tokens_size], token);
        (*tokens_size)++;
    }
    tokens[*tokens_size] = NULL; 
}

int main(int argc, char* argv[])
{
    int mode = (argc - 1);  // mode 0: interactive, 1: batch

    // ERROR 
    if(argc > 2){
        char error_message[30] = "An error has occurred\n";
        write(STDERR_FILENO, error_message, strlen(error_message)); 
        exit(1);
    }

    FILE *fp;
    if (mode == 0){
        fp = stdin;
    } 
    // else if (mode == 1){
    else {
        fp = fopen(argv[1], "r");
        if(fp == NULL){
            char error_message[30] = "An error has occurred\n";
            write(STDERR_FILENO, error_message, strlen(error_message)); 
            exit(1);
        }
    }
    
    history = (char**)malloc(MAX_HISTORY_ROW_COUNT * sizeof(char*));
    paths = (char**)malloc(MAX_PATH_COUNT * sizeof(char*));
    paths[0] = (char*)malloc(strlen(default_path) * sizeof(char));
    strcpy(paths[0], default_path);
    path_count++;

    while (1){
        char* raw_command;
        size_t bufsize = 1000;
        
        if (mode == 0){
            printf("wish> "); //print the prompt
            fflush(stdout);
        }

        raw_command = malloc(bufsize * sizeof(char));
        if(getline(&raw_command, &bufsize, fp) < 0)
            exit(0);
        if (strcmp(raw_command, "\n") == 0){
            free(raw_command);
            continue;
        }

        history[history_row_count] = (char*)malloc(bufsize * sizeof(char));
        strcpy(history[history_row_count], raw_command);
        history_row_count++;
        if (history_row_count == MAX_HISTORY_ROW_COUNT){
            char error_message[50] = "history: r_count reached MAX_HISTORY_ROW_COUNT\n";
            write(STDERR_FILENO, error_message, strlen(error_message));
            exit(1);
        }

        int *op_code = malloc(sizeof(int));
        /*  ERROR  */
        getOperationCode(raw_command, op_code);
        if (*op_code == ERROR){
            char error_message[30] = "An error has occurred\n";
            write(STDERR_FILENO, error_message, strlen(error_message));
            free(raw_command);
            free(op_code);
            continue;
        }

        char *first_tokens[50];
        int *first_tokens_size = malloc(sizeof(int));
        /*  NORMAL  */
        if (*op_code == NORMAL){                         
            parseRawCommand(strdup(raw_command), first_tokens, first_tokens_size);
            if (*first_tokens_size > 0){
                executeCommand(first_tokens, first_tokens_size, NULL);
            }
            // else {
            //     char error_message[30] = "An error has occurred\n";
            //     write(STDERR_FILENO, error_message, strlen(error_message)); 
            // }
            free(raw_command);
            // TODO: free tokens
            free(first_tokens_size);
            free(op_code);
            continue;
        }

        char *pair_command[2];
        int *pair_command_size = malloc(sizeof(int));
        splitIntoTwoCommands(strdup(raw_command), pair_command, pair_command_size);
        if (*pair_command_size != 2){
            char error_message[30] = "An error has occurred\n";
            write(STDERR_FILENO, error_message, strlen(error_message));
            free(raw_command);
            free(first_tokens_size);
            free(op_code);
            // TODO: free pair_command
            free(pair_command_size);
            continue;
        }
        // printf("REDIR and PIPE are being implemented ...\n");

        char *second_tokens[50];
        int *second_tokens_size = malloc(sizeof(int));
        /*  REDIR  */
        if (*op_code == REDIR){    
            // printf("REDIR is being implemented ...\n");
            parseRawCommand(strdup(pair_command[0]), first_tokens, first_tokens_size);
            parseRawCommand(strdup(pair_command[1]), second_tokens, second_tokens_size);
            if ((*first_tokens_size > 0) && (*second_tokens_size == 1)){
                // void executeCommand(char *tokens[50], int* tokens_size, char* redir_path)
                executeCommand(first_tokens, first_tokens_size, second_tokens[0]);
            } else {
                char error_message[30] = "An error has occurred\n";
                write(STDERR_FILENO, error_message, strlen(error_message)); 
            }
            free(raw_command);
            // TODO: free 1st tokens
            free(first_tokens_size);
            // TODO: free 2nd tokens
            free(second_tokens_size);
            // TODO: free pair_command
            free(pair_command_size);
            free(op_code);
            continue;
        }

        // printf("PIPE is being implemented ...\n");
        /* PIPE */
        parseRawCommand(strdup(pair_command[0]), first_tokens, first_tokens_size);
        parseRawCommand(strdup(pair_command[1]), second_tokens, second_tokens_size);
        executePipedCommands(first_tokens, second_tokens);
        // free
        free(raw_command);
        // TODO: free 1st tokens
        free(first_tokens_size);
        // TODO: free 2nd tokens
        free(second_tokens_size);
        // TODO: free pair_command
        free(pair_command_size);
        free(op_code);
    }


    exit(0);
}