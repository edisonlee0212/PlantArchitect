#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API GeneralTreeParameters : public IDataComponent {
        void Save(const std::filesystem::path &path) const;

        void Load(const std::filesystem::path &path);

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
        glm::vec2 m_apicalControlBaseAge;
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




        void OnInspect();

        GeneralTreeParameters();
    };

/*
     * The auxin that controls the bud flush probability, relates to apical dominance.
     */
    struct PLANT_ARCHITECT_API InternodeStatus : public IDataComponent {
        int m_age = 0;
        float m_apicalControl = 0;
        float m_inhibitor = 0;
        float m_level = 0;
        float m_distanceToRoot = 0;
        float m_maxDistanceToAnyBranchEnd = 0;
        float m_totalDistanceToAllBranchEnds = 0;
        float m_order = 0;
        float m_biomass = 0;
        float m_childTotalBiomass = 0;
        glm::quat m_desiredLocalRotation;
        float m_sagging;
        /**
         * Is child with largest total distance to all branchEnds
         */
        Entity m_largestChild = Entity();
        /**
         * Is child with largest max distance to any branch end
         */
        Entity m_longestChild = Entity();
        /**
         * Is child with largest total biomass
         */
        Entity m_heaviestChild = Entity();

        void OnInspect();
        void CalculateApicalControl(const glm::vec2 parameters, int rootAge);
    };

    /**
     * If water pressure is negative, meaning that the internode needs more water but it's empty. The pressure is controlled by apical control.
     */
    struct PLANT_ARCHITECT_API InternodeWaterPressure : public IDataComponent {
        float m_value = 0;

        void OnInspect();
    };

    /**
     * The internode water keep track of the amount of water carried by the internode.
     */
    struct PLANT_ARCHITECT_API InternodeWater : public IDataComponent {
        float m_value = 0;

        void OnInspect();
    };

    /**
     * The illumination status of the internode.
     */
    struct PLANT_ARCHITECT_API InternodeIllumination : public IDataComponent {
        float m_intensity = 0;
        glm::vec3 m_direction = glm::vec3(0.0f, 1.0f, 0.0f);

        void OnInspect();
    };
}