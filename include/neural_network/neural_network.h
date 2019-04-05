
#include <stdint.h>
#include <vector>
#include <istream>

namespace neural_network {

using Value  = float;
using Values = std::vector<Value>;



using ActivationFunction = Value (*)(Value);

Value ActivationLinear (Value value);
Value ActivationThreshold (Value value);
Value ActivationSigmoid (Value value);


class Network {
public:
    Network (size_t inputSize, std::initializer_list<size_t> hiddenSizes, size_t outputSize);
    Network (size_t inputSize, std::initializer_list<std::initializer_list<Value>> weights);
    Network (const Network &) = default;
    Network (Network &&) = default;

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

} // namespace neural_network