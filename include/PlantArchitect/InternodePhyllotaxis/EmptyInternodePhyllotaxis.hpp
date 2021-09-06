#pragma once
#include <plant_architect_export.h>
#include "IInternodePhyllotaxis.hpp"
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API EmptyInternodePhyllotaxis : public IInternodePhyllotaxis {
    public:
        void GenerateFoliage(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                             const GlobalTransform &relativeGlobalTransform) override;
    };
}