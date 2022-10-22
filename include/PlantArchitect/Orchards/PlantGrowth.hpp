#pragma once
#include "PlantStructure.hpp"
using namespace UniEngine;
namespace Orchards {
    enum class BudType{
        Shoot,
        Leaf,
        Fruit
    };

    struct BudData{

    };

    struct InternodeData{
        int m_age = 0;
        float m_inhibitor = 0;
        float m_level = 0;
        glm::quat m_desiredLocalRotation;
        float m_sagging = 0;

        int m_maxDistanceToAnyBranchEnd = 0;
        float m_order = 0;
        float m_childTotalBiomass = 0;

        int m_rootDistance = 0;
    };

    struct BranchData{

    };
    class TreeGrowthParameters {
    public:
        int m_lateralBudCount;
        /**
        * The mean and variance of the angle between the direction of a lateral bud and its parent shoot.
        */
        glm::vec2 m_branchingAngleMeanVariance;
        /**
        * The mean and variance of an angular difference orientation of lateral buds between two internodes
        */
        glm::vec2 m_rollAngleMeanVariance;
        /**
        * The mean and variance of the angular difference between the growth direction and the direction of the apical bud
        */
        glm::vec2 m_apicalAngleMeanVariance;
        float m_gravitropism;
        float m_phototropism;
        glm::vec2 m_internodeLengthMeanVariance;

        glm::vec2 m_endNodeThicknessAndControl;
        float m_lateralBudFlushingProbability;
        float m_apicalControl;
        /**
         * Avoidance multiplier, strength, max avoidance (which will completely stop bud from flushing).
         */
        glm::vec3 m_neighborAvoidance;

        /**
        * How much inhibitor will an internode generate.
        */
        glm::vec3 m_apicalDominanceBaseAgeDist;


        float m_lateralBudFlushingLightingFactor;

        glm::vec2 m_budKillProbabilityApicalLateral;

        /**
        * The minimum order of the internode that will have random pruning.
        */
        int m_randomPruningOrderProtection;
        /**
        * The base probability of an end internode being cut off due to
        * unknown environmental factors.
        */
        glm::vec3 m_randomPruningBaseAgeMax;
        /**
        * The limit of lateral branches being cut off when too close to the
        * root.
        */
        float m_lowBranchPruning;
        /**
         * The strength of gravity bending.
         */
        glm::vec3 m_saggingFactorThicknessReductionMax = glm::vec3(0.8f, 1.75f, 1.0f);

        int m_matureAge = 0;

        TreeGrowthParameters();
    };
    class TreeGrowthModel{

    public:
        TreeGrowthParameters m_parameters;
        std::shared_ptr<Plant<BranchData, InternodeData, BudData>> m_targetPlant;
        void Grow();
    };
}