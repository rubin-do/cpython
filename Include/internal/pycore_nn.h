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

	float *v_input_activations;
	float *v_hidden_activations;
	float *v_outputs;

	float *a_input_activations;
	float *a_hidden_activations;
	float *a_outputs;
} DuelingNetwork;

void init_dueling_network(DuelingNetwork* net, int n_actions, int inp_size, int hidden_size);
void dueling_forward(DuelingNetwork* net, float* x, float* q_out, int train);
void free_dueling_network(DuelingNetwork* net);
int sample_action(float* qvalues, float epsilon);

void dueling_backward(DuelingNetwork* net, int action, float predicted_q, float target_q, float learning_rate);
float compute_td_loss(
	float* state,          
    int action,            
    float reward,          
    float* next_state,     
    DuelingNetwork* agent,         
    DuelingNetwork* target_agent,  
    float gamma); 

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_NN_H */
