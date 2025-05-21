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

void dueling_forward(DuelingNetwork* net, float* x, float* q_out, int train) {
    float* V = (float*)malloc(sizeof(float));
    float* A = (float*)malloc(net->n_actions * sizeof(float));
    
    float* v_temp1 = (float*)malloc(net->V_mlp.input_layer.output_size * sizeof(float));
    float* v_temp2 = (float*)malloc(net->V_mlp.hidden_layer.output_size * sizeof(float));
    
    forward(&net->V_mlp.input_layer, x, v_temp1);
    for (int i = 0; i < net->V_mlp.input_layer.output_size; i++)
        v_temp1[i] = ReLU(v_temp1[i]);
    
    forward(&net->V_mlp.hidden_layer, v_temp1, v_temp2);
    for (int i = 0; i < net->V_mlp.hidden_layer.output_size; i++)
        v_temp2[i] = ReLU(v_temp2[i]);
    
    forward(&net->V_mlp.output_layer, v_temp2, V);

    float *a_temp1 = (float*)malloc(net->A_mlp.input_layer.output_size * sizeof(float));
    float *a_temp2 = (float*)malloc(net->A_mlp.hidden_layer.output_size * sizeof(float));
    
    forward(&net->A_mlp.input_layer, x, a_temp1);
    for (int i = 0; i < net->A_mlp.input_layer.output_size; i++)
        a_temp1[i] = ReLU(a_temp1[i]);
    
    forward(&net->A_mlp.hidden_layer, a_temp1, a_temp2);
    for (int i = 0; i < net->A_mlp.hidden_layer.output_size; i++)
        a_temp2[i] = ReLU(a_temp2[i]);
    
    forward(&net->A_mlp.output_layer, a_temp2, A);
    
    float mean_A = 0.0f;
    for (int i = 0; i < net->n_actions; i++)
        mean_A += A[i];
    mean_A /= net->n_actions;
    
    for (int i = 0; i < net->n_actions; i++)
        q_out[i] = *V + (A[i] - mean_A);

    if (train) {
	  net->v_outputs = V;
	  net->v_hidden_activations = v_temp2;
	  net->v_input_activations = v_temp1;
	  net->a_outputs = A;
	  net->a_hidden_activations = a_temp2;
	  net->a_input_activations = a_temp1;
	} else {
	  free(v_temp1);
	  free(v_temp2);
	  free(V);
	  free(a_temp1);
	  free(a_temp2);
	  free(A);
	}
}

int sample_action(float* qvalues, float epsilon) {
  float x = (float)rand()/(float)(RAND_MAX);
  if (x < epsilon) {
	return rand()%2;
  }
  return qvalues[1] > qvalues[0] ? 1 : 0;
}

void free_dueling_network(DuelingNetwork* net) {
    free_mlp(&net->V_mlp);
    free_mlp(&net->A_mlp);
}

float compute_td_loss(
	float* state,          
    int action,            
    float reward,          
    float* next_state,     
    DuelingNetwork* agent,         
    DuelingNetwork* target_agent,  
    float gamma            
) {
    float qvalues[agent->n_actions];
    dueling_forward(agent, state, qvalues, 0);
    float predicted_q = qvalues[action];

    float next_qvalues_agent[agent->n_actions];
    float next_qvalues_target[agent->n_actions];
    dueling_forward(target_agent, next_state, next_qvalues_target, 0);
    dueling_forward(agent, next_state, next_qvalues_agent, 1);

    int best_next_action = 0;
    for (int a = 1; a < agent->n_actions; a++) {
        if (next_qvalues_agent[a] > next_qvalues_agent[best_next_action]) {
            best_next_action = a;
        }
    }

    float next_state_value = next_qvalues_target[best_next_action];

    float target_q = reward + gamma * next_state_value;
    dueling_backward(agent, action, predicted_q, target_q, LR);
    
    float diff = target_q - predicted_q;
    float loss = diff * diff;

    return loss;
}

void dueling_backward(DuelingNetwork* net, int action, float predicted_q, float target_q, float learning_rate) {
    float delta_L = 2.0f * (predicted_q - target_q);

    float grad_V = delta_L;
    float *v_delta_input = (float*)malloc(net->V_mlp.input_layer.output_size * sizeof(float));
    float *v_delta_hidden = (float*)malloc(net->V_mlp.hidden_layer.output_size * sizeof(float));
    float *v_delta_output = (float*)malloc(net->V_mlp.output_layer.output_size * sizeof(float));
    float expected_V[1] = { grad_V };
    backward(&net->V_mlp, NULL, net->v_input_activations, net->v_hidden_activations, net->v_outputs, expected_V, v_delta_input, v_delta_hidden, v_delta_output);

    float inv_n_actions = 1.0f / net->n_actions;
    float grad_A[net->n_actions];
    float *a_delta_input = (float*)malloc(net->A_mlp.input_layer.output_size * sizeof(float));
    float *a_delta_hidden = (float*)malloc(net->A_mlp.hidden_layer.output_size * sizeof(float));
    float *a_delta_output = (float*)malloc(net->A_mlp.output_layer.output_size * sizeof(float));
    for (int i = 0; i < net->n_actions; i++) {
        grad_A[i] = delta_L * ((i == action) ? 1 : 0) - delta_L * inv_n_actions;
    }
    backward(&net->A_mlp, NULL, net->a_input_activations, net->a_hidden_activations, net->a_outputs, grad_A, a_delta_input, a_delta_hidden, a_delta_output);

    free(v_delta_input);
    free(v_delta_output);
    free(v_delta_hidden);
    free(a_delta_input);
    free(a_delta_output);
    free(a_delta_hidden);

	free(net->v_input_activations);
	free(net->v_hidden_activations);
	free(net->v_outputs);
	free(net->a_input_activations);
	free(net->a_hidden_activations);
	free(net->a_outputs);
}
