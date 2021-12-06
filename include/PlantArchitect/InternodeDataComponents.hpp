#pragma once
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
    /*
     * Describe the current basic structural information. Will be used for mesh generation.
     */
    struct PLANT_ARCHITECT_API InternodeInfo : public IDataComponent {
        bool m_isRealRoot = false;
        /**
         * The thickness of the internode
         */
        float m_thickness = 1.0f;
        /**
         * The length of the internode
         */
        float m_length = 0;
        /*
         * Whether this node is end node.
         */
        bool m_endNode = true;
        /*
         * The local rotation of the internode
         */
        glm::quat m_localRotation = glm::vec3(0.0f);
        /*
         * The neighbors proximity. Used against the dense crown.
         */
        float m_neighborsProximity = 0.0f;
        /**
         * The layer of internode.
         */
        unsigned m_layer = 0;
        void OnInspect();
    };
    struct PLANT_ARCHITECT_API InternodeStatistics : public IDataComponent {
        int m_lSystemStringIndex = 0;
        int m_strahlerOrder = 0;
        int m_hortonOrdering = 0;
        void OnInspect();
    };
    struct PLANT_ARCHITECT_API BranchPhysicsParameters : public IDataComponent{
#pragma region Physics
        float m_density = 1.0f;
        float m_linearDamping = 2.0f;
        float m_angularDamping = 2.0f;
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

#pragma region Debug rendering

    struct PLANT_ARCHITECT_API BranchColor : IDataComponent {
        glm::vec4 m_value;
        void OnInspect();
    };

    struct PLANT_ARCHITECT_API BranchCylinder : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const BranchCylinder &other) const {
            return other.m_value == m_value;
        }
    };

    struct PLANT_ARCHITECT_API BranchCylinderWidth : IDataComponent {
        float m_value;

        bool operator==(const BranchCylinderWidth &other) const {
            return other.m_value == m_value;
        }
        void OnInspect();
    };

    struct PLANT_ARCHITECT_API BranchPointer : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const BranchCylinder &other) const {
            return other.m_value == m_value;
        }
    };

#pragma endregion
}