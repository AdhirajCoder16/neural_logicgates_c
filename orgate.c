#include <stdio.h>
#include <math.h>

// 1. The Activation Function (Sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main() {
    // 2. Training Data (4 examples, 2 inputs each)
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double targets[4] = {0, 1, 1, 1}; // The correct answers

    // 3. Initialize Weights and Bias (starting with random-ish small numbers)
    double w1 = 0.5;
    double w2 = -0.2;
    double bias = 0.1;
    double learning_rate = 0.1;

    printf("--- Training Starting ---\n");

    // 4. The Training Loop (Run 10,000 times so it can learn)
    for (int epoch = 0; epoch < 10000; epoch++) {
        
        // Loop through all 4 training examples
        for (int i = 0; i < 4; i++) {
            double x1 = inputs[i][0];
            double x2 = inputs[i][1];
            double target = targets[i];

            // FORWARD PASS: Calculate the weighted sum and apply sigmoid
            double sum = (x1 * w1) + (x2 * w2) + bias;
            double output = sigmoid(sum);

            // BACKPROPAGATION: Calculate the error
            double error = target - output;

            // GRADIENT DESCENT: Adjust the weights and bias based on the error
            // (The "output * (1.0 - output)" is the calculus derivative part)
            double adjustment = error * output * (1.0 - output);

            w1 += learning_rate * adjustment * x1;
            w2 += learning_rate * adjustment * x2;
            bias += learning_rate * adjustment;
        }
    }

    printf("--- Training Complete ---\n\n");

    // 5. TEST THE NETWORK
    printf("Testing the network:\n");
    for (int i = 0; i < 4; i++) {
        double x1 = inputs[i][0];
        double x2 = inputs[i][1];
        
        double sum = (x1 * w1) + (x2 * w2) + bias;
        double output = sigmoid(sum);

        // %0.f rounds the output to 0 or 1 for easy reading
        printf("Input: %.0f, %.0f -> Predicted: %.4f (Rounded: %.0f)\n", 
               x1, x2, output, round(output));
    }

    return 0;
}