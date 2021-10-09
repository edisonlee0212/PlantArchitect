//
// Created by lllll on 9/1/2021.
//

#include "AutoTreeGenerationPipeline.hpp"
#include "EditorManager.hpp"
#include "GeneralTreeBehaviour.hpp"
#include "LSystemBehaviour.hpp"
#include "SpaceColonizationBehaviour.hpp"

using namespace Scripts;

void AutoTreeGenerationPipeline::Clone(const std::shared_ptr<IPrivateComponent> &target) {

}
std::shared_ptr<IInternodeBehaviour> AutoTreeGenerationPipeline::GetBehaviour(){
    return m_currentInternodeBehaviour;
}
void AutoTreeGenerationPipeline::Update() {
    auto pipelineBehaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
    if (pipelineBehaviour && m_currentInternodeBehaviour) {
        switch (m_status) {
            case AutoTreeGenerationPipelineStatus::Idle:
                pipelineBehaviour->OnIdle(*this);
                break;
            case AutoTreeGenerationPipelineStatus::BeforeGrowth:
                switch(m_behaviourType){
                    case BehaviourType::GeneralTree:
                        pipelineBehaviour->m_currentGrowingTree = std::dynamic_pointer_cast<GeneralTreeBehaviour>(m_currentInternodeBehaviour)->NewPlant(m_generalTreeParameters, Transform());
                        break;
                    case BehaviourType::LSystem:
                        pipelineBehaviour->m_currentGrowingTree = std::dynamic_pointer_cast<LSystemBehaviour>(m_currentInternodeBehaviour)->FormPlant(m_lString.Get<LString>(), m_lSystemParameters);
                        break;
                    case BehaviourType::SpaceColonization:
                        pipelineBehaviour->m_currentGrowingTree = std::dynamic_pointer_cast<SpaceColonizationBehaviour>(m_currentInternodeBehaviour)->NewPlant(m_spaceColonizationParameters, Transform());
                        break;
                }
                pipelineBehaviour->OnBeforeGrowth(*this);
                if (m_status != AutoTreeGenerationPipelineStatus::BeforeGrowth) {
                    if (!pipelineBehaviour->m_currentGrowingTree.IsValid() ||
                        !m_currentInternodeBehaviour->InternodeCheck(pipelineBehaviour->m_currentGrowingTree)) {
                        UNIENGINE_ERROR("No tree created or wrongly created!");
                        m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
                    }
                }
                break;
            case AutoTreeGenerationPipelineStatus::Growth: {
                auto internodeSystem = EntityManager::GetSystem<InternodeSystem>(EntityManager::GetCurrentScene());
                internodeSystem->Simulate(pipelineBehaviour->m_perTreeGrowthIteration);
                m_currentInternodeBehaviour->GenerateSkinnedMeshes();
                m_status = AutoTreeGenerationPipelineStatus::AfterGrowth;
                pipelineBehaviour->m_skipCurrentFrame = true;
            }
                break;
            case AutoTreeGenerationPipelineStatus::AfterGrowth:
                if (pipelineBehaviour->m_skipCurrentFrame) {
                    pipelineBehaviour->m_skipCurrentFrame = false;
                } else {
                    pipelineBehaviour->OnAfterGrowth(*this);
                    if (m_status != AutoTreeGenerationPipelineStatus::AfterGrowth) {
                        if (pipelineBehaviour->m_currentGrowingTree.IsValid())
                            m_currentInternodeBehaviour->Recycle(pipelineBehaviour->m_currentGrowingTree);
                    }
                }
                break;
        }
    }
}

static const char *BehaviourTypes[]{"GeneralTree", "LSystem", "SpaceColonization"};


void AutoTreeGenerationPipeline::OnInspect() {
    DropBehaviourButton();
    int behaviourType = (int) m_behaviourType;
    if (ImGui::Combo(
            "Internode behaviour type",
            &behaviourType,
            BehaviourTypes,
            IM_ARRAYSIZE(BehaviourTypes))) {
        m_behaviourType = (BehaviourType) behaviourType;
        UpdateInternodeBehaviour();
    }
    auto behaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
    if (behaviour) {
        ImGui::DragInt("Growth iteration", &behaviour->m_perTreeGrowthIteration);
        switch(m_behaviourType){
            case BehaviourType::GeneralTree:
            {
                ImGui::Text("General tree settings");
                FileUtils::OpenFile("Load parameters", "GeneralTreeParam", {".gtparams"}, [&](const std::filesystem::path &path) {
                    m_generalTreeParameters.Load(path);
                    m_parameterFileName = path.stem().string();
                    behaviour->m_perTreeGrowthIteration = m_generalTreeParameters.m_matureAge;

                }, false);
                FileUtils::SaveFile("Save parameters", "GeneralTreeParam", {".gtparams"}, [&](const std::filesystem::path &path) {
                    m_generalTreeParameters.Save(path);
                }, false);
                if (ImGui::TreeNodeEx("Parameters")) {
                    m_generalTreeParameters.OnInspect();
                    ImGui::TreePop();
                }
            }
                break;
            case BehaviourType::LSystem:

                break;
            case BehaviourType::SpaceColonization:
                ImGui::Text("Space colonization tree settings");
                FileUtils::OpenFile("Load parameters", "SpaceColonizationTreeParam", {".scparams"}, [&](const std::filesystem::path &path) {
                    m_spaceColonizationParameters.Load(path);
                    m_parameterFileName = path.stem().string();
                }, false);
                FileUtils::SaveFile("Save parameters", "SpaceColonizationTreeParam", {".scparams"}, [&](const std::filesystem::path &path) {
                    m_spaceColonizationParameters.Save(path);
                }, false);
                if (ImGui::TreeNodeEx("Parameters")) {
                    m_spaceColonizationParameters.OnInspect();
                    ImGui::TreePop();
                }
                break;
        }
    }


}

void AutoTreeGenerationPipeline::DropBehaviourButton() {
    if (m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>()) {
        auto behaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
        ImGui::Text("Current attached behaviour: ");
        ImGui::Button((behaviour->m_name).c_str());
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
            EditorManager::GetInstance().m_inspectingAsset = behaviour;
        }
        const std::string tag =
                "##" + behaviour->GetTypeName() + (behaviour ? std::to_string(behaviour->GetHandle()) : "");
        if (ImGui::BeginPopupContextItem(tag.c_str())) {
            if (ImGui::Button(("Remove" + tag).c_str())) {
                m_pipelineBehaviour.Clear();
            }
            ImGui::EndPopup();
        }
    } else {
        EditorManager::DragAndDropButton(m_pipelineBehaviour, "Pipeline behaviour",
                                         {"SpaceColonizationTreeToLString", "MultipleAngleCapture",
                                          "GeneralTreeToString"}, false);
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

void AutoTreeGenerationPipeline::UpdateInternodeBehaviour() {
    switch (m_behaviourType) {
        case BehaviourType::GeneralTree:
            m_currentInternodeBehaviour = std::dynamic_pointer_cast<IInternodeBehaviour>(
                    EntityManager::GetSystem<InternodeSystem>(
                            EntityManager::GetCurrentScene())->GetInternodeBehaviour<GeneralTreeBehaviour>());
            break;
        case BehaviourType::LSystem:
            m_currentInternodeBehaviour = std::dynamic_pointer_cast<IInternodeBehaviour>(
                    EntityManager::GetSystem<InternodeSystem>(
                            EntityManager::GetCurrentScene())->GetInternodeBehaviour<LSystemBehaviour>());
            break;
        case BehaviourType::SpaceColonization:
            m_currentInternodeBehaviour = std::dynamic_pointer_cast<IInternodeBehaviour>(
                    EntityManager::GetSystem<InternodeSystem>(
                            EntityManager::GetCurrentScene())->GetInternodeBehaviour<SpaceColonizationBehaviour>());
            break;
    }
}

void AutoTreeGenerationPipeline::Start() {
    UpdateInternodeBehaviour();
}

BehaviourType AutoTreeGenerationPipeline::GetBehaviourType() {
    return m_behaviourType;
}

void IAutoTreeGenerationPipelineBehaviour::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
}

void IAutoTreeGenerationPipelineBehaviour::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
}

void IAutoTreeGenerationPipelineBehaviour::OnIdle(AutoTreeGenerationPipeline &pipeline) {
}
