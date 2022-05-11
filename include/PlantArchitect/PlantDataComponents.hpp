#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API RootInfo : public IDataComponent {

    };
    struct PLANT_ARCHITECT_API BranchInfo : public IDataComponent {
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
        void OnInspect();
    };

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

        float m_rootDistance = 0;


        float m_maxDistanceToAnyBranchEnd = 0;
        float m_totalDistanceToAllBranchEnds = 0;
        float m_order = 0;
        float m_biomass = 0;
        float m_childTotalBiomass = 0;
        /**
         * Is child with largest total distance to all branch ends
         */
        bool m_largestChild = false;
        /**
         * Is child with largest max distance to any branch end
         */
        bool m_longestChild = false;
        /**
         * Is child with largest total biomass
         */
        bool m_heaviestChild = false;
        void OnInspect();
    };

    struct PLANT_ARCHITECT_API InternodeStatistics : public IDataComponent {
        int m_lSystemStringIndex = 0;
        int m_strahlerOrder = 0;
        int m_hortonOrdering = 0;
        int m_childCount = 0;

        void OnInspect();
    };


#pragma region Debug rendering

    struct PLANT_ARCHITECT_API InternodeColor : IDataComponent {
        glm::vec4 m_value;

        void OnInspect();
    };

    struct PLANT_ARCHITECT_API InternodeCylinder : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const InternodeCylinder &other) const {
            return other.m_value == m_value;
        }
    };

    struct PLANT_ARCHITECT_API InternodeCylinderWidth : IDataComponent {
        float m_value;

        bool operator==(const InternodeCylinderWidth &other) const {
            return other.m_value == m_value;
        }

        void OnInspect();
    };

    struct PLANT_ARCHITECT_API InternodePointer : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const InternodePointer &other) const {
            return other.m_value == m_value;
        }
    };

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

#pragma endregion
}