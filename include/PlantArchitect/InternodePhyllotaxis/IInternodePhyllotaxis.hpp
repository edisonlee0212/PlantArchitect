#pragma once

#include <plant_architect_export.h>
#include "Application.hpp"
#include "Internode.hpp"
#include "InternodeManager.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API IInternodePhyllotaxis : public IAsset {
    public:
        virtual void GenerateFoliage(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                                     const GlobalTransform &relativeGlobalTransform, const GlobalTransform &relativeParentGlobalTransform) = 0;
    };
}