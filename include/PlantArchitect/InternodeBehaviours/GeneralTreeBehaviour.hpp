#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API GeneralTreeTag : public IDataComponent {

    };

    struct PLANT_ARCHITECT_API GeneralTreeParameters : public IDataComponent {
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

        float m_thicknessFactor = 0.5f;
        float m_endNodeThickness;

        float m_lateralBudFlushingLightingFactor;

        float m_apicalBudKillProbability;
        float m_lateralBudKillProbability;

        /**
         * The minimum order of the internode that will have random pruning.
         */
        int m_randomPruningOrderProtection;
        /**
         * The base probability of an end internode being cut off due to
         * unknown environmental factors.
         */
        float m_randomPruningFactor;
        /**
         * How much the probability of an end internode being cut off due to
         * unknown environmental factors will increase due to internode aging.
         */
        float m_randomPruningAgeFactor;
        /**
         * The maximum probability of an end internode being cut off due to
         * unknown environmental factors will increase due to internode aging.
         */
        float m_randomPruningMax;
        /**
         * The limit of lateral branches being cut off when too close to the
         * root.
         */
        float m_lowBranchPruning;
        void OnInspect();
        GeneralTreeParameters();
    };

    /*
     * The auxin that controls the bud flush probability, relates to apical dominance.
     */
    struct PLANT_ARCHITECT_API InternodeStatus : public IDataComponent {
        float m_distanceToRoot = 0;
        float m_maxDistanceToAnyBranchEnd = 0;
        float m_totalDistanceToAllBranchEnds = 0;
        float m_order = 0;
        float m_biomass = 0;
        float m_childTotalBiomass = 0;
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
    struct PLANT_ARCHITECT_API InternodeIllumination : public IDataComponent{
        float m_intensity = 0;
        glm::vec3 m_direction = glm::vec3(0.0f, 1.0f, 0.0f);
        void OnInspect();
    };

    class PLANT_ARCHITECT_API InternodeWaterFeeder : public IPrivateComponent{
    public:
        float m_waterPerIteration = 1.0f;
        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
        void OnInspect() override;
    };



    class PLANT_ARCHITECT_API GeneralTreeBehaviour : public IInternodeBehaviour {
        std::vector<Entity> m_currentPlants;

    protected:
        bool InternalInternodeCheck(const Entity &target) override;

    public:
        void OnInspect() override;

        void OnCreate() override;

        void Grow(int iterations) override;

        Entity Retrieve() override;

        Entity Retrieve(const Entity &parent) override;

        Entity NewPlant(const GeneralTreeParameters &params, const Transform &transform);
    };
}