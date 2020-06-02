## **Implementation**

In each getline iteration,
determine the command belong to which one of the types:

+ #define ERROR -1  // invalid operation 
+ #define NORMAL 0  // valid operation without redirection nor pipping
+ #define REDIR 1   // valid redirection
+ #define PIPE 2    // valid pipping

and do corresponding execution.
