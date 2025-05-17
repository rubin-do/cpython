// This implements reward fetcher for RL garbage collector

#include "Python.h"

#include <stdio.h>

#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_memory_state.h"
#include "pycore_mlp.h"
#include "pycore_initconfig.h"    // _PyStatus_OK()

#define FILENAME "/tmp/pipe1"

void
_supervisor_routine(void* arg)
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

      fprintf(stderr, "Read reward: %f\n", value);

	  float* val = _PyMemoryState_GetEmbeddings();
	  // TODO: pass reward to model
	}

    if (feof(fp)) {
      fprintf(stderr, "End of FIFO reached.\n");
	  fflush(stderr);
    } else if (ferror(fp)) {
      perror("Error reading from FIFO");
	  fflush(stderr);
    }

    fclose(fp);
}

PyStatus
_PyGCSupervisor_Init(PyInterpreterState *interp)
{
    // Disable auto GC - supervisor will run PyGC_Collect on its own
    PyGC_Disable();

    // Run supervisor routine in separate thread
    PyThread_start_new_thread(_supervisor_routine, 0);
    return _PyStatus_OK();
}
