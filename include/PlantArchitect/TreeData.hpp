#pragma once

#include <plant_architect_export.h>

#include <TreeParameters.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API TreeData : public IPrivateComponent {
    public:
#pragma region Info
        TreeParameters m_parameters;
        float m_height;
        int m_maxBranchingDepth;
        int m_lateralBudsCount;
        float m_totalLength = 0;
#pragma endregion
#pragma region Runtime Data
        float m_activeLength = 0.0f;
        AssetRef m_branchMesh;
        AssetRef m_skinnedBranchMesh;

        bool m_meshGenerated = false;
        bool m_foliageGenerated = false;
        glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
#pragma endregion

        void OnGui() override;

        void ExportModel(const std::string &filename,
                         const bool &includeFoliage = true) const;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;

        void OnCreate() override;

        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
    };
} // namespace PlantFactory
