#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Activation Function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of Sigmoid for Backpropagation
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

int main() {
    // XOR Inputs and Targets
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double targets[4] = {0, 1, 1, 0}; // XOR: 1 only if inputs are different

    // 1. INITIALIZE WEIGHTS AND BIASES
    // Weights from Input Layer to Hidden Layer (2 inputs -> 2 hidden neurons)
    double w_input_hidden[2][2] = {
        {0.5, -0.3},  // Hidden Neuron 1 weights
        {0.1,  0.8}   // Hidden Neuron 2 weights
    };
    double bias_hidden[2] = {0.1, -0.2};

    // Weights from Hidden Layer to Output Layer (2 hidden neurons -> 1 output)
    double w_hidden_output[2] = {0.4, -0.5};
    double bias_output = 0.3;

    double learning_rate = 0.2; // Slightly higher learning rate for faster convergence

    printf("--- Training XOR Network Starting ---\n");

    // 2. THE TRAINING LOOP
    for (int epoch = 0; epoch < 30000; epoch++) { // XOR takes more epochs to learn
        for (int i = 0; i < 4; i++) {
            double x1 = inputs[i][0];
            double x2 = inputs[i][1];
            double target = targets[i];

            // --- STEP A: FORWARD PASS ---
            // Calculate Hidden Layer Outputs
            double hidden_sum1 = (x1 * w_input_hidden[0][0]) + (x2 * w_input_hidden[0][1]) + bias_hidden[0];
            double hidden_out1 = sigmoid(hidden_sum1);

            double hidden_sum2 = (x1 * w_input_hidden[1][0]) + (x2 * w_input_hidden[1][1]) + bias_hidden[1];
            double hidden_out2 = sigmoid(hidden_sum2);

            // Calculate Final Output Layer Output
            double output_sum = (hidden_out1 * w_hidden_output[0]) + (hidden_out2 * w_hidden_output[1]) + bias_output;
            double final_output = sigmoid(output_sum);


            // --- STEP B: BACKPROPAGATION ---
            // 1. Calculate error at the Output Layer
            double output_error = target - final_output;
            double output_adjustment = output_error * sigmoid_derivative(final_output);

            // 2. Pass error backward to Hidden Layer (Chain Rule)
            double hidden_error1 = output_adjustment * w_hidden_output[0];
            double hidden_adjustment1 = hidden_error1 * sigmoid_derivative(hidden_out1);

            double hidden_error2 = output_adjustment * w_hidden_output[1];
            double hidden_adjustment2 = hidden_error2 * sigmoid_derivative(hidden_out2);


            // --- STEP C: GRADIENT DESCENT (UPDATE WEIGHTS) ---
            // Update Output Layer Weights
            w_hidden_output[0] += learning_rate * output_adjustment * hidden_out1;
            w_hidden_output[1] += learning_rate * output_adjustment * hidden_out2;
            bias_output        += learning_rate * output_adjustment;

            // Update Hidden Layer Weights
            w_input_hidden[0][0] += learning_rate * hidden_adjustment1 * x1;
            w_input_hidden[0][1] += learning_rate * hidden_adjustment1 * x2;
            bias_hidden[0]       += learning_rate * hidden_adjustment1;

            w_input_hidden[1][0] += learning_rate * hidden_adjustment2 * x1;
            w_input_hidden[1][1] += learning_rate * hidden_adjustment2 * x2;
            bias_hidden[1]       += learning_rate * hidden_adjustment2;
        }
    }

    printf("--- Training Complete ---\n\n");

    // 3. TEST THE XOR NETWORK
    printf("Testing the XOR network:\n");
    for (int i = 0; i < 4; i++) {
        double x1 = inputs[i][0];
        double x2 = inputs[i][1];
        
        // Forward pass one last time to get the trained answers
        double hidden_out1 = sigmoid((x1 * w_input_hidden[0][0]) + (x2 * w_input_hidden[0][1]) + bias_hidden[0]);
        double hidden_out2 = sigmoid((x1 * w_input_hidden[1][0]) + (x2 * w_input_hidden[1][1]) + bias_hidden[1]);
        double final_output = sigmoid((hidden_out1 * w_hidden_output[0]) + (hidden_out2 * w_hidden_output[1]) + bias_output);

        printf("Input: %.0f, %.0f -> Predicted Probability: %.4f (Rounded: %.0f)\n", 
               x1, x2, final_output, round(final_output));
    }

    return 0;
}