// This implements gc supervisor for RL garbage collector

#include "Python.h"

#include <stdio.h>
#include <fcntl.h>  

#include "pycore_pystate.h" // _PyThreadState_GET()
#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_memory_state.h"
#include "pycore_mlp.h"
#include "pycore_initconfig.h"    // _PyStatus_OK()

#define FILENAME "/tmp/pipe1"

void
_PyGCSupervisor_Run()
{
  // Get fp from thread state
  PyThreadState *tstate = _PyThreadState_GET();

//  /* Only execute pending calls on the main thread. */
//  if (!_Py_IsMainThread() || !_Py_IsMainInterpreter(tstate->interp)) {
//	return;
//  }

  FILE* fp = (FILE*)tstate->reward_file;
  if (fp == NULL) {
	// file is not initialized yet
	return;
  }

  char buffer[1024];

  // Try to read from fd
  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
	  char *endptr;
	  errno = 0; 

      float value = strtof(buffer, &endptr);

      if (endptr == buffer) {
        fprintf(stderr, "Warning: No float found in line: %s", buffer);
      } else if (errno == ERANGE) {
        fprintf(stderr, "Warning: Float out of range in line: %s", buffer);
      }

      fprintf(stderr, "Read reward: %f\n", value);
	  fflush(stderr);

	  //float* val = _PyMemoryState_GetEmbeddings();
	  // TODO: pass reward to model
	}
}

PyStatus
_PyGCSupervisor_Init(PyThreadState *tstate)
{
    // Disable auto GC - supervisor will run PyGC_Collect on its own
    PyGC_Disable();

	int fd = open(FILENAME, O_RDWR | O_NONBLOCK);
	if (fd == -1) {
	  perror("Error opening file");
	  return _PyStatus_ERR("Error opening file");
	}

	FILE *fp;
	fp = fdopen(fd, "r");

	if (fp == NULL) {
	  perror("Error opening FIFO");
	  return _PyStatus_ERR("Error opening FIFO");
	}
  
    // init thread state
    tstate->reward_file = (uintptr_t)fp;

    return _PyStatus_OK();
}
