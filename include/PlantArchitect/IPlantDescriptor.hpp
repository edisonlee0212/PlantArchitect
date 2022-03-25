#pragma once
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
    class IPlantDescriptor : public IAsset{
    public:
        virtual Entity InstantiateTree() = 0;

        void OnInspect() override;
    };
}