// This implements reward fetcher for RL garbage collector

#include "Python.h"

#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_initconfig.h"    // _PyStatus_OK()

PyStatus
_PyRewardFetcher_Init(PyInterpreterState *interp)
{
    return _PyStatus_OK();
}
