// This implements memory state embeddings getter

#include "Python.h"

#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_initconfig.h"    // _PyStatus_OK()

float* _PyMemoryState_GetEmbeddings();

float* _PyMemoryState_GetEmbeddings() {
  return 0;
}
