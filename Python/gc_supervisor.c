// This implements gc supervisor for RL garbage collector

#include "Python.h"

#include <stdio.h>
#include <fcntl.h>  
#include <time.h>

#include "pycore_pystate.h" // _PyThreadState_GET()
#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_memory_state.h"
#include "pycore_nn.h"
#include "pycore_initconfig.h"    // _PyStatus_OK()

#define FILENAME "/tmp/pipe1"
#define METRICS_FILENAME "/tmp/metrics"

// TODO: fix input dim
#define N_INPUTS 64
#define N_HIDDEN 64
#define N_OUTPUTS 2

#define COPY_MODEL_ITERATIONS 20
#define METRICS_ITERATIONS 20

#define EPS 0.07

int iter = 0;

// metrics
double last_inference_time = 0;
int gc_calls = 0;

void dump_metrics(FILE *metrics_file) {
  fprintf(metrics_file, "%d %f\n", gc_calls, last_inference_time);
  fflush(metrics_file);
  last_inference_time = 0;
  gc_calls = 0;
}

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

  FILE* metrics_file = (FILE*)tstate->metrics_file;
  if (metrics_file == NULL) {
    return;
  }

  char buffer[1024];
  float qvalues[2];

  // Try to read from fd
  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
	  char *endptr;
	  errno = 0; 

      clock_t start = clock();

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

      clock_t end = clock();

      last_inference_time = (double)(end-start) / CLOCKS_PER_SEC;

	  if (action) {
        gc_calls++;
		PyGC_Collect();
	  }

	  float* next_state = _PyMemoryState_GetEmbeddings();

	  DuelingNetwork* target_net = (DuelingNetwork*)tstate->dueling_target_nn;
	  compute_td_loss(state, action, reward, next_state, net, target_net, 0.99f);

	  if (!(iter % COPY_MODEL_ITERATIONS)) {
		copy_dueling_network(target_net, net);
	  }

      if (!(iter % METRICS_ITERATIONS)) {
        dump_metrics(metrics_file);
        iter = 0;
      }
	
	  free(state);
	  free(next_state);

	  fprintf(stderr, "Qvalues: %f %f\n", qvalues[0], qvalues[1]);

	  fflush(stderr);
	  iter++;
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

    FILE *metrics_file;
    metrics_file = fopen(METRICS_FILENAME, "w");

    DuelingNetwork *net = malloc(sizeof(DuelingNetwork));
    init_dueling_network(net, N_OUTPUTS, N_INPUTS, N_HIDDEN);

    DuelingNetwork *target_net = malloc(sizeof(DuelingNetwork));
    init_dueling_network(target_net, N_OUTPUTS, N_INPUTS, N_HIDDEN);
  
    // init thread state
    tstate->reward_file = (uintptr_t)fp;
    tstate->metrics_file = (uintptr_t)metrics_file;
    tstate->dueling_nn = (uintptr_t)net;
    tstate->dueling_target_nn = (uintptr_t)target_net;

    return _PyStatus_OK();
}
