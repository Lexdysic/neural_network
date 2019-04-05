#include <neural_network/neural_network.h>

#include <numeric>
#include <cassert>

namespace neural_network {

Value ActivationLinear (Value value) {
    return value;
}

Value ActivationThreshold (Value value) {
    return value > Value(0) ? Value(1) : Value(0);
}

Value ActivationSigmoid (Value value) {
    if (value < Value(-45.0))
        return Value(0);
    if (value > Value(45.0))
        return Value(1);
    return Value(1) / (Value(1) + std::exp(-value));
}


Network::Network (
    size_t                        inputSize,
    std::initializer_list<size_t> hiddenSizes, 
    size_t                        outputSize
)
    : m_inputSize(inputSize)
    , m_layers(hiddenSizes.size() + size_t(1))
    , m_activation(ActivationThreshold)
{
    size_t prevSize = inputSize;
    auto initLayer = [&prevSize](Layer & layer, size_t size) {
        assert(size > size_t(0));
        layer.neurons.resize(size);
        layer.weights.resize(size * prevSize);

        prevSize = size;
    };

    auto layerIt = m_layers.begin();
    for (size_t size : hiddenSizes) {
        initLayer(*layerIt, size);
    }

    initLayer(m_layers.back(), outputSize);
}

Network::Network (size_t inputSize, std::initializer_list<std::initializer_list<Value>> layers)
    : m_inputSize(inputSize)
    , m_layers(layers.size())
    , m_activation(ActivationThreshold)
{
    size_t prevSize = inputSize;
    auto initLayer = [&prevSize](Layer & layer, std::initializer_list<Value> weights) {
        size_t size = weights.size() / prevSize;

        layer.neurons.resize(size);
        layer.weights.assign(weights);

        prevSize = size;
    };

    auto layerIt = m_layers.begin();
    for (auto weights : layers) {
        initLayer(*layerIt++, weights);
    }
}

void Network::Run (const Values & inputs) {
    assert(inputs.size() == m_inputSize);

    const Values * previous = &inputs;

    for (auto & layer : m_layers) {
        auto weights = layer.weights.cbegin();
        for (auto & neuron : layer.neurons) {
            const Value value = std::inner_product(
                previous->cbegin(),
                previous->cend(),
                weights,
                layer.bias
            );

            neuron = m_activation(value);

            std::advance(weights, previous->size());
        }

        assert(weights == layer.weights.end());

        previous = &layer.neurons;
    }

}

void Network::Assign (std::initializer_list<std::initializer_list<Value>> layers) {
    size_t prevSize = m_inputSize;
    auto assignLayer = [&prevSize](Layer & layer, std::initializer_list<Value> weights) {
        size_t neuronSize = weights.size() / prevSize;
        assert(weights.size() == layer.weights.size());
        assert(neuronSize == layer.neurons.size());

        layer.neurons.resize(neuronSize);
        layer.weights.assign(weights);

        prevSize = neuronSize;
    };

    auto layerIt = m_layers.begin();
    for (auto weights : layers) {
        assignLayer(*layerIt++, weights);
    }
}

const Values & Network::GetOutput () const {
    return m_layers.back().neurons;
}

} // namespace neural_network