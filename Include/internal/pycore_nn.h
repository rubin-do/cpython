#ifndef Py_INTERNAL_NN_H
#define Py_INTERNAL_NN_H
#include "pycore_mlp.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

typedef struct {
    MLP V_mlp;
    MLP A_mlp;
    int n_actions;
} DuelingNetwork;

void init_dueling_network(DuelingNetwork* net, int n_actions, int inp_size, int hidden_size);
void dueling_forward(DuelingNetwork* net, float* x, float* q_out);
void free_dueling_network(DuelingNetwork* net);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_NN_H */
