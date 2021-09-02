#pragma once
#include <InternodeSystem.hpp>

using namespace PlantArchitect;

namespace Scripts {
    enum class AutoTreeGenerationPipelineStatus{
        Idle,
        BeforeGrowth,
        Growth,
        AfterGrowth
    };
    class AutoTreeGenerationPipeline : public IPrivateComponent {
        void DropBehaviourButton();

    public:
        AutoTreeGenerationPipelineStatus m_status = AutoTreeGenerationPipelineStatus::Idle;
        AssetRef m_pipelineBehaviour;
        void Update() override;
        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
        void OnGui() override;

    };

    class IAutoTreeGenerationPipelineBehaviour : public IAsset{
    public:
        virtual void OnIdle(AutoTreeGenerationPipeline& pipeline);
        virtual void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline);
        virtual void OnGrowth(AutoTreeGenerationPipeline& pipeline);
        virtual void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline);
    };
}