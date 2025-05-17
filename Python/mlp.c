// #include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// TODO: fix input dim
#define N_INPUTS 64
#define N_HIDDEN 64
#define N_OUTPUTS 2;

// learning rate
#define LR 0.01

typedef union {float f; int32_t i;} fandi;

int32_t ftoi(float f){fandi f2i; f2i.f = f; return f2i.i;}
float itof(int32_t i){fandi i2f; i2f.i = i; return i2f.f;}
// branchless ReLU: https://gist.github.com/ToruNiina/f7a3ba69585cf3bfd869e302357c11a8
float ReLU(float x)
{
    // s, e, f mean `sign bit`, `exponent`, and `fractional`, respectively.
    //     seeeffff |     seeeffff
    //     00010100 |     10010100 --+
    // >>        31 |  >>       31   |
    // -------------+-------------   |
    // not 00000000 | not 11111111   |
    // -------------+-------------   |
    //     11111111 |     00000000   |
    // and 00010100 | and 10010100 <-+
    // -------------+-------------
    //     00010100 |     00000000
    // ==         x |          0.0
    const int32_t xi = ftoi(x);
    return itof(xi & ~(xi >> 31));
}

float ReLU_derivative(float x) {
    if (x >= 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}


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

// TODO: move declarations to Include/internal 
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

void init_layer(LinearLayer *layer, int input_size, int output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;

    layer->weights = (float **)malloc(input_size * sizeof(float *));
    for (int i = 0; i < input_size; i++) {
        layer->weights[i] = (float *)malloc(output_size * sizeof(float));
    }

    layer->biases = (float *)malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
        for (int j = 0; j < output_size; j++)
            layer->weights[i][j] = ((float)rand() / RAND_MAX) - 0.5;

    for (int i = 0; i < output_size; i++)
        layer->biases[i] = ((float)rand() / RAND_MAX) - 0.5;
}

void free_layer(LinearLayer *layer) {
    for (int i = 0; i < layer->input_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
}

void init_mlp(MLP *mlp) {
  srand(time(NULL));
  init_layer(&mlp->input_layer, N_INPUTS, N_HIDDEN);
  init_layer(&mlp->hidden_layer, N_HIDDEN, N_HIDDEN);
  init_layer(&mlp->output_layer, N_HIDDEN, N_OUTPUTS);
}

void free_mlp(MLP *mlp) {
  free_layer(&mlp->input_layer);
  free_layer(&mlp->hidden_layer);
  free_layer(&mlp->output_layer);
}

void forward(LinearLayer* layer, float inputs[], float outputs[]) {
      for (int i = 0; i < layer->output_size; i++) {
        float activation = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            activation += inputs[j] * layer->weights[j][i];
        }
        
        outputs[i] = ReLU(activation);
    }
}

void backward(MLP *mlp,
              float inputs[],
              float input_outputs[],
              float hidden_outputs[],
              float output_outputs[],
			  float expected_outputs[],
			  float delta_input[],
			  float delta_hidden[],
			  float delta_output[]) {
    for (int i = 0; i < nn->output_layer.output_size; i++) {
        delta_output[i] = errors[i] * sigmoid_derivative(output_outputs[i]);
    }

    for (int i = 0; i < nn->hidden_layer.output_size; i++) {
        float error = 0.0;
        for (int j = 0; j < nn->output_layer.output_size; j++) {
            error += delta_output[j] * nn->output_layer.weights[i][j];
        }
        delta_hidden[i] = error * sigmoid_derivative(hidden_outputs[i]);
    }

    for (int i = 0; i < nn->input_layer.output_size; i++) {
        float error = 0.0;
        for (int j = 0; j < nn->hidden_layer.output_size; j++) {
            error += delta_hidden[j] * nn->hidden_layer.weights[i][j];
        }
        delta_input[i] = error * ReLU_derivative(input_outputs[i]);
    }
}

void update(LinearLayer* layer, float inputs[], float deltas[]) {
    for (int i = 0; i < layer->input_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            layer->weights[i][j] += LR * deltas[j] * inputs[i];
        }
    }

    for (int i = 0; i < layer->output_size; i++) {
        layer->biases[i] += LR * deltas[i];
    }
}
