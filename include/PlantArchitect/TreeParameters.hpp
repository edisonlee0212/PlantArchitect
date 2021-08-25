#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API TreeParameters {
    public:
        void Serialize(YAML::Emitter &out);

        void Deserialize(const YAML::Node &in);

        void OnGui();

        void Serialize(std::filesystem::path path) const;

        void Deserialize(std::filesystem::path path);

#pragma region Parameters
        /**
         * \brief How many lateral bud will an internode generate.
         */
        int m_lateralBudPerNode = 2;
#pragma region Geometric
        /**
         * \brief The average of apical angle.
         */
        float m_apicalAngleMean = 15.0f;
        /**
         * \brief The variance of apical angle.
         */
        float m_apicalAngleVariance = 1.0f;
        /**
         * \brief The average of branching angle.
         */
        float m_branchingAngleMean = 30;
        /**
         * \brief The variance of branching angle.
         */
        float m_branchingAngleVariance = 2;
        /**
         * \brief The average of roll angle.
         */
        float m_rollAngleMean = 60;
        /**
         * \brief The variance of roll angle.
         */
        float m_rollAngleVariance = 1;
        /**
         * \brief The distance between two nearest internodes.
         */
        float m_internodeLengthBase = 1;
#pragma endregion
#pragma region Bud fate
        /**
         * \brief How much carbon does an apical bud needs to flush.
         */
        float m_apicalIlluminationRequirement = 1;
        /**
         * \brief How much carbon does a lateral bud needs to flush.
         */
        float m_lateralIlluminationRequirement = 1;
        /**
         * \brief How much inhibitor will an internode generate.
         */
        float m_inhibitorBase = 0.9f;
        /**
         * \brief How much inhibitor will be preserved when being transited to parent
         * internodes.
         */
        float m_inhibitorDistanceFactor = 0.8f;
        /**
         * \brief How much resource will an apical bud receive compare with lateral
         * buds from resource allocation
         */
        float m_resourceWeightApical = 1.0f;
        /**
         * \brief The variance of weight between apical and lateral buds for resource
         * allocation.
         */
        float m_resourceWeightVariance = 0.0f;
        /**
         * \brief How much will the resource allocation favor to main child. (Child
         * with higher level)
         */
        float m_apicalControlLevelFactor = 0.9f;

        /**
         * \brief The minimum resource for a plant will grow.
         */
        float m_heightResourceHeightDecreaseMin = 0.01f;

        /**
         * \brief How resource decrease as tree grow higher.
         */
        float m_heightResourceHeightDecreaseBase = 0.0f;
        /**
         * \brief The speed of how resource decrease as tree grow higher.
         */
        float m_heightResourceHeightDecreaseFactor = 1.0f;

#pragma endregion
#pragma region Environmental
        /**
         * \brief The avoidance angle to prevent self-collision for close branches.
         */
        float m_avoidanceAngle = 10.0f;
        /**
         * \brief How much will an internode rotate towards/against the light
         * direction.
         */
        float m_phototropism = 0.0f;
        /**
         * \brief How much will an internode rotate towards/against the direction of
         * gravity.
         */
        float m_gravitropism = 0.1f;
        /**
         * \brief The base probability of an end internode being cut off due to
         * unknown environmental factors.
         */
        float m_randomCutOff = -0.02f;
        /**
         * \brief How much the probability of an end internode being cut off due to
         * unknown environmental factors will increase due to internode aging.
         */
        float m_randomCutOffAgeFactor = 0.002f;
        /**
         * \brief The maximum probability of an end internode being cut off due to
         * unknown environmental factors will increase due to internode aging.
         */
        float m_randomCutOffMax = 0.03f;
        /**
         * \brief The limit of lateral branches being cut off when too close to the
         * root.
         */
        float m_lowBranchCutOff = 0.1f;
#pragma endregion
        /**
         * \brief The thickness of the end node.
         */
        float m_endNodeThickness = 0.01f;
        /**
         * \brief The control factor thickness for thickness calculation.
         */
        float m_thicknessControlFactor = 1.0f;
        /**
         * \brief The strength of gravity bending.
         */
        float m_gravityBendingFactor = 0.8f;
        /**
         * \brief The strength of a branch fight against gravity bending with its
         * thickness.
         */
        float m_gravityBendingThicknessFactor = 1.75f;
        /**
         * \brief The maximum bending strength of an internode.
         */
        float m_gravityBendingMax = 1.0f;
#pragma region Organs
        /**
         * \brief The type of this tree.
         */
        int m_treeType = 0;
#pragma endregion
#pragma endregion
    };
} // namespace PlantFactory
