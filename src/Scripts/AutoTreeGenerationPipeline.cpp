//
// Created by lllll on 9/1/2021.
//

#include "AutoTreeGenerationPipeline.hpp"
#include "EditorManager.hpp"
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

void AutoTreeGenerationPipeline::OnInspect() {
    DropBehaviourButton();
}

void AutoTreeGenerationPipeline::DropBehaviourButton() {
    if(m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>()){
        auto behaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
        ImGui::Text("Current attached behaviour: ");
        ImGui::Button((behaviour->m_name).c_str());
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
        {
            EditorManager::GetInstance().m_inspectingAsset = behaviour;
        }
        const std::string tag = "##" + behaviour->GetTypeName() + (behaviour ? std::to_string(behaviour->GetHandle()) : "");
        if (ImGui::BeginPopupContextItem(tag.c_str()))
        {
            if (ImGui::Button(("Remove" + tag).c_str()))
            {
                m_pipelineBehaviour.Clear();
            }
            ImGui::EndPopup();
        }
    }else {
        EditorManager::DragAndDropButton(m_pipelineBehaviour, "Pipeline behaviour",
                                         {"SpaceColonizationTreeToLString", "MultipleAngleCapture", "GeneralTreeToString"}, false);
    }
}

void AutoTreeGenerationPipeline::Serialize(YAML::Emitter &out) {
    m_pipelineBehaviour.Save("m_pipelineBehaviour", out);
}

void AutoTreeGenerationPipeline::Deserialize(const YAML::Node &in) {
    m_pipelineBehaviour.Load("m_pipelineBehaviour", in);
}

void AutoTreeGenerationPipeline::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_pipelineBehaviour);
}



void IAutoTreeGenerationPipelineBehaviour::OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) {

}

void IAutoTreeGenerationPipelineBehaviour::OnGrowth(AutoTreeGenerationPipeline& pipeline) {

}

void IAutoTreeGenerationPipelineBehaviour::OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) {

}

void IAutoTreeGenerationPipelineBehaviour::OnIdle(AutoTreeGenerationPipeline &pipeline) {

}
