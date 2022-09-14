#pragma once

#include "Application.hpp"
#include "plant_architect_export.h"
#include "ILayer.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API StrandLayer : public ILayer {
    public:
        void OnCreate() override;

        void OnInspect() override;
    };
}
