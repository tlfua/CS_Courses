#include "types.h"
#include "stat.h"
#include "user.h"
#include "fs.h"
#include "fcntl.h"
#include "syscall.h"
#include "traps.h"


int
main(int argc, char *argv[])
{

  // printf(1, "%s\n", argv[0]);
  printf(1, "befor experiments, there are %d opens\n", getopenedcount());

  int k = atoi(argv[1]);
  printf(1, "experiment to open file %d times\n", k);

  for (int i=0; i<k; i++){
    int fd = open("test_opened", O_CREATE|O_RDWR);
    if(fd < 0){
      printf(1, "error: creat big failed!\n");
      exit();
    }
    // close(fd);
  }

  printf(1, "after experiment, there are %d opens\n", getopenedcount());

  exit();
}
