#ifndef Py_INTERNAL_GC_H
#define Py_INTERNAL_GC_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

float* _PyMemoryState_GetEmbeddings(void);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_GC_H */
