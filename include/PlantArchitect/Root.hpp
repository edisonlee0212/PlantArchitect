#pragma once

#include "InternodeRingSegment.hpp"
#include <plant_architect_export.h>
#include <IInternodeResource.hpp>
#include <PlantDataComponents.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API BranchPhysicsParameters {
#pragma region Physics
        float m_density = 1.0f;
        float m_linearDamping = 4.0f;
        float m_angularDamping = 4.0f;
        int m_positionSolverIteration = 8;
        int m_velocitySolverIteration = 8;
        float m_jointDriveStiffnessFactor = 3000.0f;
        float m_jointDriveStiffnessThicknessFactor = 4.0f;
        float m_jointDriveDampingFactor = 10.0f;
        float m_jointDriveDampingThicknessFactor = 4.0f;
        bool m_enableAccelerationForDrive = true;
#pragma endregion

        void Serialize(YAML::Emitter &out);

        void Deserialize(const YAML::Node &in);

        void OnInspect();
    };

    class PLANT_ARCHITECT_API Root : public IPrivateComponent {
    public:
        glm::vec3 m_center;
        BranchPhysicsParameters m_branchPhysicsParameters;
        void OnInspect() override;
        void OnCreate() override;
        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}