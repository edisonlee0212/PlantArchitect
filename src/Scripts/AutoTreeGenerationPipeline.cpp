//
// Created by lllll on 9/1/2021.
//

#include "AutoTreeGenerationPipeline.hpp"
using namespace Scripts;
void AutoTreeGenerationPipeline::Clone(const std::shared_ptr<IPrivateComponent> &target) {

}

void AutoTreeGenerationPipeline::Update() {
    auto behaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
    if(behaviour){
        switch (m_status) {
            case AutoTreeGenerationPipelineStatus::Idle:
                behaviour->OnIdle(*this);
                break;
            case AutoTreeGenerationPipelineStatus::BeforeGrowth:
                behaviour->OnBeforeGrowth(*this);
                break;
            case AutoTreeGenerationPipelineStatus::Growth:
                behaviour->OnGrowth(*this);
                break;
            case AutoTreeGenerationPipelineStatus::AfterGrowth:
                behaviour->OnAfterGrowth(*this);
                break;
        }
    }
}

void IAutoTreeGenerationPipelineBehaviour::OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) {

}

void IAutoTreeGenerationPipelineBehaviour::OnIdle(AutoTreeGenerationPipeline& pipeline) {

}

void IAutoTreeGenerationPipelineBehaviour::OnGrowth(AutoTreeGenerationPipeline& pipeline) {

}

void IAutoTreeGenerationPipelineBehaviour::OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) {

}
