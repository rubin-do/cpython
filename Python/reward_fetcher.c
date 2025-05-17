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

	while (fgets(buffer, sizeof(buffer), fp) != NULL) {
	  char *endptr;
	  errno = 0; 

      float value = strtof(buffer, &endptr);

      if (endptr == buffer) {
        fprintf(stderr, "Warning: No float found in line: %s", buffer);
        continue;
      } else if (errno == ERANGE) {
        fprintf(stderr, "Warning: Float out of range in line: %s", buffer);
        continue;
      }

      printf("Read reward: %f\n", value);
	}

    if (feof(fp)) {
      printf("End of FIFO reached.\n");
	  fflush(stdout);
    } else if (ferror(fp)) {
      perror("Error reading from FIFO");
	  fflush(stdout);
    }

    fclose(fp);
}

PyStatus
_PyRewardFetcher_Init(PyInterpreterState *interp)
{
    PyThread_start_new_thread(_fetcher_routine, 0);
    return _PyStatus_OK();
}
