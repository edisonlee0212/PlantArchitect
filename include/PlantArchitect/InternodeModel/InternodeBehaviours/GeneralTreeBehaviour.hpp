#pragma once

#include <plant_architect_export.h>
#include "InternodeModel/IPlantBehaviour.hpp"
#include "InternodeModel/IPlantDescriptor.hpp"

using namespace UniEngine;
namespace PlantArchitect {
#pragma region Data Components
    struct PLANT_ARCHITECT_API GeneralTreeTag : public IDataComponent {

    };


    class PLANT_ARCHITECT_API GeneralTreeParameters : public IPlantDescriptor {
    public:
        Entity InstantiateTree() override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

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

        void OnInspect() override;
        void CollectAssetRef(std::vector<AssetRef> &list) override;
        void OnCreate() override;
    };

/*
     * The auxin that controls the bud flush probability, relates to apical dominance.
     */
    struct PLANT_ARCHITECT_API InternodeStatus : public IDataComponent {
        int m_branchingOrder = 0;
        int m_age = 0;
        int m_treeAge = 0;
        float m_gravitropism;
        float m_apicalControl = 0;
        float m_inhibitor = 0;
        float m_level = 0;
        float m_branchLength = 0;
        float m_recordedProbability = 0;
        glm::quat m_desiredLocalRotation;
        float m_sagging = 0;
        float m_currentTotalNodeCount = 0;
        float m_startDensity = 0.0f;
        float m_chainDistance = 0;

        void OnInspect();

        void CalculateApicalControl(float apicalControl);
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

#pragma endregion


    class PLANT_ARCHITECT_API InternodeWaterFeeder : public IPrivateComponent {
    public:
        float m_lastRequest = 0;
        float m_waterDividends = 1.0f;

        void OnInspect() override;
    };

    

    class PLANT_ARCHITECT_API GeneralTreeBehaviour : public IPlantBehaviour {
    protected:
        bool InternalInternodeCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;

        bool InternalRootCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;

        bool InternalBranchCheck(const std::shared_ptr<Scene> &scene, const Entity &target) override;

        void
        CalculateChainDistance(const std::shared_ptr<Scene> &scene, const Entity &target, float previousChainDistance);

    public:
        void Preprocess(const std::shared_ptr<Scene> &scene, std::vector<Entity> &currentRoots);

        void OnMenu() override;

        GeneralTreeBehaviour();

        void Grow(const std::shared_ptr<Scene> &scene, int iteration) override;

        Entity CreateRoot(const std::shared_ptr<Scene> &scene, AssetRef descriptor, Entity &rootInternode,
                          Entity &rootBranch) override;

        Entity
        CreateBranch(const std::shared_ptr<Scene> &scene, const Entity &parent, const Entity &internode) override;

        Entity CreateInternode(const std::shared_ptr<Scene> &scene, const Entity &parent) override;

        Entity NewPlant(const std::shared_ptr<Scene> &scene, const std::shared_ptr<GeneralTreeParameters> &descriptor,
                        const Transform &transform);
    };

    
}