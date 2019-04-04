
#include <stdint.h>
#include <vector>

using Weight  = float;
using Neuron  = float;
using Layer   = std::vector<Neuron>;
using Weights = std::vector<Weight>;

class NeuralNetwork {
public:
    void Run (const Layer & input);
    void Train (const Layer & input, const Layer & output);

    void Read (FILE * file);
    void Write (FILE * file);

private:
    std::vector<Layer> m_hidden;
    Layer              m_outputs;
    Layer              m_delta;
};