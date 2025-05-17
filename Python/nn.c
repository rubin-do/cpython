// This implements neural networks for RL garbage collector

#include "Python.h"
#include "pycore_nn.h"

#include <stdlib.h>
#include <math.h>

void init_dueling_network(DuelingNetwork* net, int n_actions, int inp_size, int hidden_size) {
    net->n_actions = n_actions;
    
    init_mlp(&net->V_mlp, inp_size, hidden_size, 1);
    
    init_mlp(&net->A_mlp, inp_size, hidden_size, n_actions);
}

void dueling_forward(DuelingNetwork* net, float* x, float* q_out) {
    float V;
    float* A = (float*)malloc(net->n_actions * sizeof(float));
    
    float* temp1 = (float*)malloc(net->V_mlp.input_layer.output_size * sizeof(float));
    float* temp2 = (float*)malloc(net->V_mlp.hidden_layer.output_size * sizeof(float));
    
    forward(&net->V_mlp.input_layer, x, temp1);
    for (int i = 0; i < net->V_mlp.input_layer.output_size; i++)
        temp1[i] = ReLU(temp1[i]);
    
    forward(&net->V_mlp.hidden_layer, temp1, temp2);
    for (int i = 0; i < net->V_mlp.hidden_layer.output_size; i++)
        temp2[i] = ReLU(temp2[i]);
    
    forward(&net->V_mlp.output_layer, temp2, &V);
    
    forward(&net->A_mlp.input_layer, x, temp1);
    for (int i = 0; i < net->A_mlp.input_layer.output_size; i++)
        temp1[i] = ReLU(temp1[i]);
    
    forward(&net->A_mlp.hidden_layer, temp1, temp2);
    for (int i = 0; i < net->A_mlp.hidden_layer.output_size; i++)
        temp2[i] = ReLU(temp2[i]);
    
    forward(&net->A_mlp.output_layer, temp2, A);
    
    float mean_A = 0.0f;
    for (int i = 0; i < net->n_actions; i++)
        mean_A += A[i];
    mean_A /= net->n_actions;
    
    for (int i = 0; i < net->n_actions; i++)
        q_out[i] = V + (A[i] - mean_A);
    
    free(temp1);
    free(temp2);
    free(A);
}

void free_dueling_network(DuelingNetwork* net) {
    free_mlp(&net->V_mlp);
    free_mlp(&net->A_mlp);
}
