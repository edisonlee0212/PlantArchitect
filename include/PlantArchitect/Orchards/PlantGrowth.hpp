#pragma once

#include "PlantStructure.hpp"

using namespace UniEngine;
namespace Orchards {
    enum class BudType {
        Apical,
        LateralVegetative,
        LateralReproductive
    };

    enum class BudStatus {
        Dormant,
        Flushed,
        Died
    };

    class Bud {
    public:
        BudType m_type = BudType::Apical;
        BudStatus m_status = BudStatus::Dormant;
        glm::quat m_localRotation = glm::vec3(0.0f);
    };
    struct InternodeData {
        int m_age = 0;
        float m_inhibitor = 0;
        glm::quat m_desiredLocalRotation = glm::vec3(0.0f);
        float m_sagging = 0;

        float m_maxDistanceToAnyBranchEnd = 0;
        float m_level = 0;
        float m_childTotalBiomass = 0;

        float m_rootDistance = 0;

        float m_apicalControl = 0.0f;
        int m_decedentsAmount = 0;
        glm::vec3 m_lightDirection = glm::vec3(0, 1, 0);
        float m_lightIntensity = 1.0f;

        /**
         * List of buds, first one will always be the apical bud which points forward.
         */
        std::vector<Bud> m_buds;
        void Clear();
    };

    struct BranchData {
        int m_order = 0;
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
        float m_internodeLength;
        float m_growthRate;
        glm::vec2 m_endNodeThicknessAndControl;
        float m_lateralBudFlushingProbability;
        /*
         * To form significant trunk. Larger than 1 means forming big trunk.
         */
        glm::vec2 m_apicalControlBaseDistFactor;
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

    struct GrowthNutrients{
        float m_water = 0.0f;
    };

    class TreeGrowthModel {
        bool m_initialized = false;
        void CalculateSagging(InternodeHandle internodeHandle);
        void CollectInhibitor(InternodeHandle internodeHandle);
        void GrowInternode(InternodeHandle internodeHandle, const GrowthNutrients& growthNutrients);
    public:
        glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
        TreeGrowthParameters m_parameters;
        std::shared_ptr<Plant<BranchData, InternodeData>> m_targetPlant;
        void Initialize();
        void Clear();
        void Grow(const GrowthNutrients& growthNutrients);
    };
}