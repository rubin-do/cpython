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
    if (append(state->objects, (uintptr_t)obj) != 0) {
	  return 0; // stop iteration
	}
    if (append(state->refcnts, (size_t)Py_REFCNT(obj)) != 0) {
	  return 0;
	}
    return 1;
}

#define OUTPUT 64

float* get_embeddings(Array *arr1, Array* arr2) {
    float *embeddings = (float*)malloc(sizeof(float) * OUTPUT);
    if (!embeddings) return NULL;

    memset(embeddings, 0, sizeof(float) * OUTPUT);

    size_t total_len = arr1->size + arr2->size;
    float normalization_factor = (total_len > 0) ? (1.0f / (float)total_len) : 0.0f;

    for (size_t i = 0; i < arr1->size; i++) {
        embeddings[i % OUTPUT] += ((float)arr1->array[i]) * normalization_factor;
    }

    for (size_t i = 0; i < arr2->size; i++) {
        embeddings[i % OUTPUT] += ((float)arr2->array[i]) * normalization_factor;
    }

    return embeddings;
}

float* _PyMemoryState_GetEmbeddings() {
    Array objs = {NULL, 0, 0};
    Array refcnts = {NULL, 0, 0};
    MemoryState state = {&objs, &refcnts};

    PyUnstable_GC_VisitObjects(gcvisitobjects_callback, &state);
    float *embeddings = get_embeddings(state.objects, state.refcnts);

#ifdef Py_DEBUG
    fprintf(stderr, "visited %d objects\n", objs.size);
	for (size_t i = 0; i < OUTPUT; i++) {
	  fprintf(stderr, "%f ", embeddings[i]); 
	}
    fprintf(stderr, "\n");
    fflush(stderr);
#endif

    // Free allocated memory when done
    free(objs.array);
    free(refcnts.array);

    return embeddings;
}

