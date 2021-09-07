#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class Internode;

    struct InternodeInfo;

    class PLANT_ARCHITECT_API InternodeFoliage : public IAsset {
    public:
        AssetRef m_foliagePhyllotaxis;

        void Generate(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                      const GlobalTransform &relativeGlobalTransform);

        void OnInspect();
    };
}