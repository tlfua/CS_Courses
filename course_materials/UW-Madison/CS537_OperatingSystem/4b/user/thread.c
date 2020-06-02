#include "types.h"
#include "stat.h"
#include "fcntl.h"
#include "user.h"
#include "x86.h"
#include "param.h"
#define PGSIZE (4096)
// #define ORG_STACKS_SIZE 500

// int count_thread_create = 0;
void* org_stacks[NPROC];

// void* addthread[NPROC];

int
thread_create(void (*start_routine)(void *, void *), void *arg1, void *arg2)
{
  void* org_stack = malloc(2*PGSIZE);
  if (org_stack == 0)
    return -1;

  void* stack = org_stack;
  uint offset = (uint)stack % PGSIZE;
  if (offset != 0)
    stack = (void*) ((uint)stack + (4096 - offset));

  int ret_clone = clone(start_routine, arg1, arg2, stack);
  if (ret_clone != -1) {
    org_stacks[ret_clone % NPROC] = org_stack;
  }

  return ret_clone;
}

int
thread_join(void)
{
  void *stack;
  int threadid = join(&stack);
  if(stack == NULL) return threadid;
  else {
    free(org_stacks[threadid % NPROC]);
  }

  return threadid;
}

