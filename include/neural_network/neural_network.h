
#include <stdint.h>
#include <vector>
#include <istream>

using Value  = float;
using Values = std::vector<Value>;

using ActivationFunction = Value (*)(Value);

class NeuralNetwork {
public:
    NeuralNetwork (size_t inputSize, std::initializer_list<size_t> hiddenSizes, size_t outputSize);
    NeuralNetwork (size_t inputSize, std::initializer_list<std::initializer_list<Value>> weights);
    NeuralNetwork (const NeuralNetwork &) = default;
    NeuralNetwork (NeuralNetwork &&) = default;

    void Run (const Values & input);
    void Assign (std::initializer_list<std::initializer_list<Value>> weights);
    //void Train (const Values & input, const Values & output);

    const Values & GetOutput () const;


private:
    struct Layer {
        Values neurons;
        Values weights;
        Value  bias = 0;
    };

    ActivationFunction m_activation;
    size_t             m_inputSize;
    std::vector<Layer> m_layers;
};