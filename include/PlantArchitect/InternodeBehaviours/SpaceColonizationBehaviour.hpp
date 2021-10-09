#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API SpaceColonizationTag : public IDataComponent {
        bool m_truck = false;
    };
    struct PLANT_ARCHITECT_API SpaceColonizationIncentive : public IDataComponent {
        glm::vec3 m_direction;
        int m_pointAmount;
    };

    struct PLANT_ARCHITECT_API SpaceColonizationParameters : public IDataComponent {
        float m_removeDistance = 0.8f;
        float m_attractDistance = 3.0f;
        float m_internodeLengthMean = 0.5f;
        float m_internodeLengthVariance = 0.1f;


        float m_thicknessFactor = 0.5f;
        float m_endNodeThickness = 0.02f;

        void OnInspect();
        void Save(const std::filesystem::path &path) const;
        void Load(const std::filesystem::path &path);
    };

    class IVolume;

    class PLANT_ARCHITECT_API SpaceColonizationBehaviour : public IInternodeBehaviour {
        void VolumeSlotButton();

    protected:
        bool InternalInternodeCheck(const Entity &target) override;

    public:
        std::vector<PrivateComponentRef> m_volumes;
        std::vector<glm::vec3> m_attractionPoints;
        glm::vec3 m_center;

        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;

        void OnInspect() override;

        void OnCreate() override;

        void Grow(int iterations) override;

        void PushVolume(const std::shared_ptr<IVolume> &volume);

        Entity Retrieve() override;

        Entity Retrieve(const Entity &parent) override;

        Entity NewPlant(const SpaceColonizationParameters &params,
                        const Transform &transform);
    };
}