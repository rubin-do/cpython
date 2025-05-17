// This implements reward fetcher for RL garbage collector

#include "Python.h"

#include <stdio.h>

#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_initconfig.h"    // _PyStatus_OK()

#define FILENAME "/tmp/pipe1"

void
_fetcher_routine(void*)
{
  FILE *fp;
  fp = fopen(FILENAME, "r");

  char buffer[1024];

  if (fp == NULL) {
	perror("Error opening FIFO");
	return;
  }

  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
	// fgets stops after reading a line or buffer limit
	printf("Read one line from FIFO: %s", buffer);
  } else {
	printf("No data read from FIFO or error occurred.\n");
  }

  printf("%s\n", buffer);
  fflush(stdout);
}

PyStatus
_PyRewardFetcher_Init(PyInterpreterState *interp)
{
    PyThread_start_new_thread(_fetcher_routine, 0);
    return _PyStatus_OK();
}
