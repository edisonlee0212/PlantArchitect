#pragma once
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
    /*
     * Describe the current basic structural information. Will be used for mesh generation.
     */
    struct PLANT_ARCHITECT_API InternodeInfo : public IDataComponent {
        /**
         * The thickness of the internode
         */
        float m_thickness = 1.0f;
        /**
         * The length of the internode
         */
        float m_length = 0;
        /**
         * The index of current internode within a plant, start from the root.
         */
        int m_index = -1;
        /**
         * The current root of the internode.
         */
        Entity m_currentRoot = Entity();
        /*
         * Whether this node is end node.
         */
        bool m_endNode = true;
        /*
         * The local rotation of the internode
         */
        glm::quat m_localRotation = glm::vec3(0.0f);
    };

#pragma region Debug rendering

    struct PLANT_ARCHITECT_API BranchColor : IDataComponent {
        glm::vec4 m_value;
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
    };

    struct PLANT_ARCHITECT_API BranchPointer : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const BranchCylinder &other) const {
            return other.m_value == m_value;
        }
    };

#pragma endregion
}