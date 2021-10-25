#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class Internode;

    struct InternodeInfo;

    class PLANT_ARCHITECT_API InternodeFoliage : public IAsset {
    public:
        AssetRef m_foliagePhyllotaxis;
        AssetRef m_foliageTexture;
        glm::vec3 m_foliageColor = glm::vec3(0, 1, 0);
        void Generate(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                      const GlobalTransform &relativeGlobalTransform);

        void OnInspect() override;
        void Serialize(YAML::Emitter &out) override;
        void CollectAssetRef(std::vector<AssetRef> &list) override;
        void Deserialize(const YAML::Node &in) override;
    };
}