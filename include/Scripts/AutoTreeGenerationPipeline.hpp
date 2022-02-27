#pragma once
#include <PlantLayer.hpp>
#include <GeneralTreeBehaviour.hpp>
#include <SpaceColonizationBehaviour.hpp>
#include <LSystemBehaviour.hpp>
using namespace PlantArchitect;

namespace Scripts {
    enum class AutoTreeGenerationPipelineStatus{
        Idle,
        BeforeGrowth,
        Growth,
        AfterGrowth
    };

    enum class BehaviourType{
        GeneralTree,
        LSystem,
        SpaceColonization
    };

    class AutoTreeGenerationPipeline : public IPrivateComponent {
        void DropBehaviourButton();

        std::shared_ptr<IPlantBehaviour> m_currentInternodeBehaviour;
        BehaviourType m_behaviourType = BehaviourType::GeneralTree;
    public:
        void UpdateInternodeBehaviour();
        std::string m_parameterFileName;
        GeneralTreeParameters m_generalTreeParameters;
        SpaceColonizationParameters m_spaceColonizationParameters;
        LSystemParameters m_lSystemParameters;
        AssetRef m_lString;
        AutoTreeGenerationPipelineStatus m_status = AutoTreeGenerationPipelineStatus::Idle;
        AssetRef m_pipelineBehaviour;

        void Update() override;
        void OnInspect() override;
        void Start() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        BehaviourType GetBehaviourType();
        std::shared_ptr<IPlantBehaviour> GetBehaviour();
        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };

    class IAutoTreeGenerationPipelineBehaviour : public IAsset{
        friend class AutoTreeGenerationPipeline;
    public:
        Entity m_currentGrowingTree;
        int m_perTreeGrowthIteration = 1;
        virtual void OnIdle(AutoTreeGenerationPipeline& pipeline);
        virtual void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline);
        virtual void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline);
    };
}