#pragma once
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
    class IPlantDescriptor : public IAsset{
    public:
        glm::vec3 m_foliageColor = glm::vec3(0, 1, 0);
        glm::vec3 m_branchColor = glm::vec3(40.0f / 255, 15.0f / 255, 0.0f);

        AssetRef m_foliagePhyllotaxis;
        AssetRef m_branchTexture;

        virtual Entity InstantiateTree() = 0;

        void OnInspect() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };
}