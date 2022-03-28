//
// Created by lllll on 9/1/2021.
//

#include "AutoTreeGenerationPipeline.hpp"
#include "Editor.hpp"
#include "GeneralTreeBehaviour.hpp"
#include "LSystemBehaviour.hpp"
#include "SpaceColonizationBehaviour.hpp"

using namespace Scripts;

std::shared_ptr<IPlantBehaviour> AutoTreeGenerationPipeline::GetBehaviour() {
    return m_currentInternodeBehaviour;
}

void AutoTreeGenerationPipeline::Update() {
    auto pipelineBehaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
    if (pipelineBehaviour && m_currentInternodeBehaviour) {
        switch (m_status) {
            case AutoTreeGenerationPipelineStatus::Idle:
                if (!m_busy) {
                    break;
                } else if (m_remainingInstanceAmount > 0) {
                    m_remainingInstanceAmount--;
                    m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
                } else if (!m_descriptors.empty()) {
                    m_currentUsingDescriptor = m_descriptors.back();
                    m_descriptors.pop_back();
                    m_remainingInstanceAmount = m_generationAmount;
                } else {
                    pipelineBehaviour->OnEnd(*this);
                    UNIENGINE_LOG("Task finished!");
                    m_busy = false;
                }
                break;
            case AutoTreeGenerationPipelineStatus::BeforeGrowth:
                switch (m_behaviourType) {
                    case BehaviourType::GeneralTree:
                        m_iterations = m_currentUsingDescriptor.Get<GeneralTreeParameters>()->m_matureAge;
                        m_prefix = "GeneralTree_" + m_currentUsingDescriptor.Get<IPlantDescriptor>()->GetName() + "_";
                        break;
                    case BehaviourType::LSystem:
                        m_prefix = "LSystemString_" + m_currentUsingDescriptor.Get<IPlantDescriptor>()->GetName() + "_";
                        break;
                    case BehaviourType::SpaceColonization:
                        m_prefix = "SpaceColonization_" + m_currentUsingDescriptor.Get<IPlantDescriptor>()->GetName() + "_";
                        break;
                }
                m_prefix += std::to_string(m_generationAmount - m_remainingInstanceAmount + m_startIndex);
                switch (m_behaviourType) {
                    case BehaviourType::GeneralTree:
                        m_currentGrowingTree = std::dynamic_pointer_cast<GeneralTreeBehaviour>(
                                m_currentInternodeBehaviour)->NewPlant(m_currentUsingDescriptor, Transform());
                        break;
                    case BehaviourType::LSystem:
                        m_currentGrowingTree = std::dynamic_pointer_cast<LSystemBehaviour>(
                                m_currentInternodeBehaviour)->NewPlant(m_currentUsingDescriptor, Transform());
                        break;
                    case BehaviourType::SpaceColonization:
                        m_currentGrowingTree = std::dynamic_pointer_cast<SpaceColonizationBehaviour>(
                                m_currentInternodeBehaviour)->NewPlant(m_currentUsingDescriptor, Transform());
                        break;
                }
                pipelineBehaviour->OnBeforeGrowth(*this);
                if (m_status != AutoTreeGenerationPipelineStatus::BeforeGrowth) {
                    if (!m_currentGrowingTree.IsValid() ||
                        !m_currentInternodeBehaviour->RootCheck(m_currentGrowingTree)) {
                        UNIENGINE_ERROR("No tree created or wrongly created!");
                        m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
                    }
                }
                break;
            case AutoTreeGenerationPipelineStatus::Growth: {
                Application::GetLayer<PlantLayer>()->Simulate(m_iterations);
                m_status = AutoTreeGenerationPipelineStatus::AfterGrowth;
            }
                break;
            case AutoTreeGenerationPipelineStatus::AfterGrowth:
                pipelineBehaviour->OnAfterGrowth(*this);
                if (m_status != AutoTreeGenerationPipelineStatus::AfterGrowth) {
                    if (m_currentGrowingTree.IsValid())
                        Entities::DeleteEntity(Entities::GetCurrentScene(), m_currentGrowingTree);
                }
                break;
        }
    }
}

static const char *BehaviourTypes[]{"GeneralTree", "LSystem", "SpaceColonization"};

void IAutoTreeGenerationPipelineBehaviour::OnStart(AutoTreeGenerationPipeline &pipeline) {

}

void AutoTreeGenerationPipeline::OnInspect() {
    ImGui::DragInt("Start Index", &m_startIndex);
    ImGui::DragInt("Amount per descriptor", &m_generationAmount);
    auto behaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
    if (!behaviour) {
        ImGui::Text("Behaviour missing!");
    } else if (m_busy) {
        ImGui::Text("Task dispatched...");
        ImGui::Text(("Remaining descriptors: " + std::to_string(m_descriptors.size())).c_str());
        ImGui::Text(("Total: " + std::to_string(m_generationAmount) + ", Remaining: " +
                     std::to_string(m_remainingInstanceAmount)).c_str());
        if (ImGui::Button("Force stop")) {
            m_remainingInstanceAmount = 0;
            m_descriptors.clear();
            m_busy = false;
        }
    } else {
        int behaviourType = (int) m_behaviourType;
        if (ImGui::Combo(
                "Plant behaviour type",
                &behaviourType,
                BehaviourTypes,
                IM_ARRAYSIZE(BehaviourTypes))) {
            SetBehaviourType((BehaviourType)behaviourType);
        }
        ImGui::Text(("Loaded descriptors: " + std::to_string(m_descriptors.size())).c_str());
        FileUtils::OpenFolder("Collect descriptors", [&](const std::filesystem::path &path) {
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                for (const auto &entry: std::filesystem::recursive_directory_iterator(path)) {
                    if (!std::filesystem::is_directory(entry.path())) {
                         switch (m_behaviourType) {
                            case BehaviourType::GeneralTree:
                                if (entry.path().extension() == ".gtparams") {
                                    auto descriptor = AssetManager::CreateAsset<GeneralTreeParameters>();
                                    descriptor->SetPathAndLoad(entry.path());
                                    m_descriptors.emplace_back(descriptor);
                                    break;
                                }
                                break;
                            case BehaviourType::LSystem:
                                if (entry.path().extension() == ".lstring") {
                                    auto descriptor = AssetManager::CreateAsset<LSystemString>();
                                    descriptor->SetPathAndLoad(entry.path());
                                    m_descriptors.emplace_back(descriptor);
                                    break;
                                }
                                break;
                            case BehaviourType::SpaceColonization:
                                if (entry.path().extension() == ".scparams") {
                                    auto descriptor = AssetManager::CreateAsset<SpaceColonizationParameters>();
                                    descriptor->SetPathAndLoad(entry.path());
                                    m_descriptors.emplace_back(descriptor);
                                    break;
                                }
                                break;
                        }
                    }
                }
            }

        });
        if(m_descriptors.empty()){
            ImGui::Text("No descriptors!");
        }
        else if (Application::IsPlaying()) {
            if (ImGui::Button("Start")) {
                m_busy = true;
                behaviour->OnStart(*this);
                m_status = AutoTreeGenerationPipelineStatus::Idle;
            }
        } else {
            ImGui::Text("Start Engine first!");
        }
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
            m_currentInternodeBehaviour = std::dynamic_pointer_cast<IPlantBehaviour>(
                    Application::GetLayer<PlantLayer>()->GetPlantBehaviour<GeneralTreeBehaviour>());
            break;
        case BehaviourType::LSystem:
            m_currentInternodeBehaviour =
                    Application::GetLayer<PlantLayer>()->GetPlantBehaviour<LSystemBehaviour>();
            break;
        case BehaviourType::SpaceColonization:
            m_currentInternodeBehaviour =
                    Application::GetLayer<PlantLayer>()->GetPlantBehaviour<SpaceColonizationBehaviour>();
            break;
    }

}

void AutoTreeGenerationPipeline::Start() {
    UpdateInternodeBehaviour();
}

BehaviourType AutoTreeGenerationPipeline::GetBehaviourType() {
    return m_behaviourType;
}

void AutoTreeGenerationPipeline::SetBehaviourType(BehaviourType type) {
    m_behaviourType = type;
    UpdateInternodeBehaviour();
    m_currentUsingDescriptor.Clear();
    m_descriptors.clear();
}

void IAutoTreeGenerationPipelineBehaviour::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
}

void IAutoTreeGenerationPipelineBehaviour::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
}

void IAutoTreeGenerationPipelineBehaviour::OnEnd(AutoTreeGenerationPipeline &pipeline) {

}
