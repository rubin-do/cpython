#ifndef Py_INTERNAL_MLP_H
#define Py_INTERNAL_MLP_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

// learning rate
#define LR 0.01

typedef union {float f; int32_t i;} fandi;

int32_t ftoi(float f);
float itof(int32_t i);

// branchless ReLU: https://gist.github.com/ToruNiina/f7a3ba69585cf3bfd869e302357c11a8
float ReLU(float x);

float ReLU_derivative(float x);

typedef struct {
  int input_size;
  int output_size;
  float** weights; // input_size x output_size
  float *biases; // output_size
} LinearLayer;

typedef struct {
  LinearLayer input_layer;
  LinearLayer hidden_layer;
  LinearLayer output_layer;
} MLP;

void init_layer(LinearLayer* layer, int input_size, int output_size);
void copy_layer(LinearLayer* dst, LinearLayer* src);
void free_layer(LinearLayer* layer);

void init_mlp(MLP *mlp, int inp_size, int hidden_size, int output_size);
void copy_mlp(MLP *dst, MLP *src);
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
