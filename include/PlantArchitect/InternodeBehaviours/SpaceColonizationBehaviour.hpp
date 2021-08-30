#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API SpaceColonizationTag : public IDataComponent {

    };
    struct PLANT_ARCHITECT_API SpaceColonizationIncentive : public IDataComponent {
        glm::vec3 m_direction;
        int m_pointAmount;
    };

    class PLANT_ARCHITECT_API SpaceColonizationBehaviour : public IInternodeBehaviour {
    public:
        std::vector<PrivateComponentRef> m_volumes;

        float m_removeDistance;
        float m_attractDistance;
        float m_internodeLength;
        std::vector<glm::vec3> m_attractionPoints;
        glm::vec3 m_center;

        void OnInspect() override;

        void OnCreate() override;

        void PreProcess() override;

        void PostProcess() override;

        void Grow() override;
    };
}