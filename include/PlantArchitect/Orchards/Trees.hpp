#pragma once

#include "plant_architect_export.h"
#include "PlantGrowth.hpp"
using namespace UniEngine;
namespace Orchards {
    class Trees : public IPrivateComponent{
    public:
        std::vector<std::pair<Transform, TreeGrowthModel>> m_trees;

        void OnInspect() override;

        void OnCreate() override;

        void OnDestroy() override;
    };
}