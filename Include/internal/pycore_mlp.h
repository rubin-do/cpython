#ifndef Py_INTERNAL_MLP_H
#define Py_INTERNAL_MLP_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

void init_layer(LinearLayer* layer, int input_size, int output_size);
void free_layer(LinearLayer* layer);

void init_mlp(MLP *mlp);
void free_mlp(MLP *mlp);

void forward(LinearLayer* layer, float inputs[], float outputs[]);
void backward(MLP *mlp,
              float inputs[],
              float input_outputs[],
              float hidden_outputs[],
              float output_outputs[],
			  float expected_outputs[],
			  float delta_input[],
			  float delta_hidden[],
			  float delta_output[]);

void update(LinearLayer* layer, float inputs[], float deltas[]);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_MLP_H */
