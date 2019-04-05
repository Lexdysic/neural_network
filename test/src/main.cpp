#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <neural_network/neural_network.h>

TEST_CASE("Simple XOR", "[run]") {
    neural_network::Network nn(2, {2}, 1);

    nn.Assign(
        {
            { 1.0f, 1.0f, 1.0f, 1.0f },
            { 1.0f, -1.0f }
        }
    );

    nn.Run({0, 0});
    CHECK(nn.GetOutput()[0] == 0.0f);

    nn.Run({0, 1});
    CHECK(nn.GetOutput()[0] == 1.0f);

    nn.Run({1, 0});
    CHECK(nn.GetOutput()[0] == 1.0f);

    nn.Run({1, 1});
    CHECK(nn.GetOutput()[0] == 0.0f);
}