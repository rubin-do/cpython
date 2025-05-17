// This implements memory state embeddings getter

#include <stdio.h>
#include <stdlib.h>

#include "Python.h"
#include "pycore_initconfig.h"    // _PyStatus_OK()
#include "pycore_memory_state.h"
#include "refcount.h" // Py_REFCNT()

typedef struct {
  size_t *array; // TODO: which type?
    size_t size;
    size_t capacity;
} Array;

typedef struct {
  Array *objects;
  Array *refcnts;
} MemoryState;

static int ensure_capacity(Array *arr) {
    if (arr->size >= arr->capacity) {
        size_t new_capacity = arr->capacity ? arr->capacity * 2 : 16;
        size_t* new_array = realloc(arr->array, new_capacity * sizeof(size_t));
        if (!new_array) return 1;  
        arr->array = new_array;
        arr->capacity = new_capacity;
    }
    return 0;
}

static int append(Array *arr, size_t n) {
    if (ensure_capacity(arr) != 0) {
        return 1;  
    }
    arr->array[arr->size++] = n;
    return 0;
}

int gcvisitobjects_callback(PyObject *obj, void *arg) {
    MemoryState *state = (MemoryState *)arg;
    if (append(state->objects, 1) != 0) {
	  return 0; // stop iteration
	}
    if (append(state->refcnts, 1) != 0) {
	  return 0;
	}
    return 1;
}

float* _PyMemoryState_GetEmbeddings() {
    Array objs = {NULL, 0, 0};
    Array refcnts = {NULL, 0, 0};
    MemoryState state = {&objs, &refcnts};

    PyUnstable_GC_VisitObjects(gcvisitobjects_callback, &state);

    // TODO: get embeds
    fprintf(stderr, "visited %d objects\n", objs.size);
    fflush(stderr);

    // Free allocated memory when done
    free(objs.array);
    free(refcnts.array);

    return 0;
}

