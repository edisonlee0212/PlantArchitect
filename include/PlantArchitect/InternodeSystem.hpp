#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API InternodeSystem : public ISystem {
    public:
        void Simulate(float delta);
    };
}