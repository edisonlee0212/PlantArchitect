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
        /**
         * The index of current internode within a plant, start from the root.
         */
        int m_index = -1;
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
        void OnInspect(){
            ImGui::Text(("Proximity: " + std::to_string(m_neighborsProximity)).c_str());

            ImGui::Text(("Thickness: " + std::to_string(m_thickness)).c_str());
            ImGui::Text(("Length: " + std::to_string(m_length)).c_str());
            ImGui::Text(("Index: " + std::to_string(m_index)).c_str());
            ImGui::Text(("Is end node: " + std::to_string(m_endNode)).c_str());

            glm::vec3 localRotation = glm::eulerAngles(m_localRotation);
            ImGui::Text(("Local Rotation: [" + std::to_string(glm::degrees(localRotation.x)) + ", " + std::to_string(glm::degrees(localRotation.y)) + ", " +std::to_string(glm::degrees(localRotation.z)) + "]").c_str());
        }
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