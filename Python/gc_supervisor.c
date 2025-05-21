// This implements gc supervisor for RL garbage collector

#include "Python.h"

#include <stdio.h>
#include <fcntl.h>  

#include "pycore_pystate.h" // _PyThreadState_GET()
#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_memory_state.h"
#include "pycore_nn.h"
#include "pycore_initconfig.h"    // _PyStatus_OK()

#define FILENAME "/tmp/pipe1"

// TODO: fix input dim
#define N_INPUTS 64
#define N_HIDDEN 64
#define N_OUTPUTS 2

#define COPY_MODEL_ITERATIONS 20

#define EPS 0.07

int infer_iter = 0;

void
_PyGCSupervisor_Run()
{
  // Get fp from thread state
  PyThreadState *tstate = _PyThreadState_GET();

  if (!_Py_IsMainThread() || !_Py_IsMainInterpreter(tstate->interp)) {
	return;
  }

  FILE* fp = (FILE*)tstate->reward_file;
  if (fp == NULL) {
	// file is not initialized yet
	return;
  }

  char buffer[1024];
  float qvalues[2];

  // Try to read from fd
  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
	  char *endptr;
	  errno = 0; 

      float reward = strtof(buffer, &endptr);

      if (endptr == buffer) {
        fprintf(stderr, "Warning: No float found in line: %s", buffer);
      } else if (errno == ERANGE) {
        fprintf(stderr, "Warning: Float out of range in line: %s", buffer);
      }

#ifdef Py_DEBUG
      fprintf(stderr, "Read reward: %f\n", reward);
	  fflush(stderr);
#endif

	  float* state = _PyMemoryState_GetEmbeddings();

	  DuelingNetwork* net = (DuelingNetwork*)tstate->dueling_nn;

	  // pass embeds to model
	  dueling_forward(net, state, qvalues, 0);

	  int action = sample_action(qvalues, EPS);
	  if (action) {
		PyGC_Collect();
	  }

	  float* next_state = _PyMemoryState_GetEmbeddings();

	  DuelingNetwork* target_net = (DuelingNetwork*)tstate->dueling_target_nn;
	  compute_td_loss(state, action, reward, next_state, net, target_net, 0.99f);

	  if (!(infer_iter % COPY_MODEL_ITERATIONS)) {
		infer_iter = 0;
		copy_dueling_network(target_net, net);
	  }
	
	  free(state);
	  free(next_state);

	  fprintf(stderr, "Qvalues: %f %f", qvalues[0], qvalues[1]);
	  fflush(stderr);
	  infer_iter++;
	  // TODO: replay buffer
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

    DuelingNetwork *net = malloc(sizeof(DuelingNetwork));
    init_dueling_network(net, N_OUTPUTS, N_INPUTS, N_HIDDEN);

    DuelingNetwork *target_net = malloc(sizeof(DuelingNetwork));
    init_dueling_network(target_net, N_OUTPUTS, N_INPUTS, N_HIDDEN);
  
    // init thread state
    tstate->reward_file = (uintptr_t)fp;
    tstate->dueling_nn = (uintptr_t)net;
    tstate->dueling_target_nn = (uintptr_t)target_net;

    return _PyStatus_OK();
}
