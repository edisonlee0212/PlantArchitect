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
        SpaceColonization,
        TreeGraph
    };

    class AutoTreeGenerationPipeline : public IPrivateComponent {
        std::shared_ptr<IPlantBehaviour> m_currentInternodeBehaviour;
    public:
        int m_startIndex = 1;
        std::string m_prefix;
        Entity m_currentGrowingTree;
        BehaviourType m_behaviourType = BehaviourType::GeneralTree;
        AssetRef m_currentUsingDescriptor;
        bool m_busy = false;
        int m_remainingInstanceAmount = 0;
        int m_generationAmount = 1;
        int m_iterations = 0;
        void UpdateInternodeBehaviour();

        std::filesystem::path m_currentDescriptorPath;

        std::vector<std::filesystem::path> m_descriptorPaths;
        AutoTreeGenerationPipelineStatus m_status = AutoTreeGenerationPipelineStatus::Idle;
        AssetRef m_pipelineBehaviour;
        void Update() override;
        void OnInspect() override;
        void Start() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
        BehaviourType GetBehaviourType();
        [[nodiscard]] int GetSeed() const;
        void SetBehaviourType(BehaviourType type);
        std::shared_ptr<IPlantBehaviour> GetBehaviour();
        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };

    class IAutoTreeGenerationPipelineBehaviour : public IAsset{
        friend class AutoTreeGenerationPipeline;
    public:
        virtual void OnStart(AutoTreeGenerationPipeline& pipeline);
        virtual void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline);
        virtual void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline);
        virtual void OnEnd(AutoTreeGenerationPipeline& pipeline);
    };
}