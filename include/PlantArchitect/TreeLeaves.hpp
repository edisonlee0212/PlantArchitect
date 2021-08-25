#pragma once

#include <plant_architect_export.h>

#include <TreeSystem.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API TreeLeaves final : public IPrivateComponent {
    public:
        std::vector<int> m_targetBoneIndices;
        std::vector<glm::mat4> m_transforms;

        AssetRef m_leavesMesh;
        AssetRef m_skinnedLeavesMesh;

        void OnGui() override;

        void FormSkinnedMesh(std::vector<unsigned> &boneIndices);

        void FormMesh();

        void OnCreate() override;

        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };
} // namespace PlantFactory
