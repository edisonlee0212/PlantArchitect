#pragma once

#include <plant_architect_export.h>
#include "IPlantBehaviour.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API SpaceColonizationTag : public IDataComponent {
        bool m_truck = false;
    };
    struct PLANT_ARCHITECT_API SpaceColonizationIncentive : public IDataComponent {
        glm::vec3 m_direction;
        int m_pointAmount;
    };

    class PLANT_ARCHITECT_API SpaceColonizationParameters : public IPlantDescriptor {
    public:
        Entity InstantiateTree() override;

        float m_removeDistance = 0.8f;
        float m_attractDistance = 3.0f;
        float m_internodeLengthMean = 0.5f;
        float m_internodeLengthVariance = 0.1f;
        float m_thicknessFactor = 0.5f;
        float m_endNodeThickness = 0.02f;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        void OnInspect() override;
    };

    class IVolume;

    class PLANT_ARCHITECT_API SpaceColonizationBehaviour : public IPlantBehaviour {
        void VolumeSlotButton();

    protected:
        bool InternalInternodeCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;
        bool InternalRootCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;
        bool InternalBranchCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;

    public:
        std::vector<PrivateComponentRef> m_volumes;
        std::vector<glm::vec3> m_attractionPoints;
        glm::vec3 m_center;

        void OnInspect() override;

        SpaceColonizationBehaviour();

        void Grow(const std::shared_ptr<Scene> &scene, int iterations) override;

        void PushVolume(const std::shared_ptr<IVolume> &volume);

        Entity CreateRoot(const std::shared_ptr<Scene> &scene, AssetRef descriptor, Entity& rootInternode, Entity& rootBranch) override;
        Entity CreateBranch(const std::shared_ptr<Scene> &scene, const Entity &parent, const Entity &internode) override;
        Entity CreateInternode(const std::shared_ptr<Scene> &scene, const Entity &parent) override;

        Entity NewPlant(const std::shared_ptr<Scene> &scene, const std::shared_ptr<SpaceColonizationParameters> &descriptor,
                        const Transform &transform);
    };
}