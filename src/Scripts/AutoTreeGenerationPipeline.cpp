//
// Created by lllll on 9/1/2021.
//

#include "AutoTreeGenerationPipeline.hpp"
#include "Editor.hpp"
#include "GeneralTreeBehaviour.hpp"
#include "LSystemBehaviour.hpp"
#include "SpaceColonizationBehaviour.hpp"
#include "TreeGraph.hpp"
using namespace Scripts;

std::shared_ptr<IPlantBehaviour> AutoTreeGenerationPipeline::GetBehaviour() {
    return m_currentInternodeBehaviour;
}

void AutoTreeGenerationPipeline::Update() {
    auto pipelineBehaviour = m_pipelineBehaviour.Get<IAutoTreeGenerationPipelineBehaviour>();
    if (pipelineBehaviour && m_currentInternodeBehaviour) {
        auto scene = GetScene();
        switch (m_status) {
            case AutoTreeGenerationPipelineStatus::Idle:
                if (!m_busy) {
                    break;
                } else if (m_remainingInstanceAmount > 0) {
                    m_remainingInstanceAmount--;
                    m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
                } else if (!m_descriptorPaths.empty()) {
                    m_currentDescriptorPath = m_descriptorPaths.back();
                    switch (m_behaviourType) {
                        case BehaviourType::GeneralTree:
                            if (m_currentDescriptorPath.m_isInProjectFolder) {
                                m_currentUsingDescriptor = std::dynamic_pointer_cast<GeneralTreeParameters>(
                                        ProjectManager::GetOrCreateAsset(m_currentDescriptorPath.m_path));
                            } else {
                                auto descriptor = ProjectManager::CreateTemporaryAsset<GeneralTreeParameters>();
                                descriptor->Import(m_currentDescriptorPath.m_path);
                                m_currentUsingDescriptor = descriptor;
                            }
                            break;
                        case BehaviourType::LSystem:
                            if (m_currentDescriptorPath.m_isInProjectFolder) {
                                m_currentUsingDescriptor = std::dynamic_pointer_cast<LSystemString>(
                                        ProjectManager::GetOrCreateAsset(m_currentDescriptorPath.m_path));
                            } else {
                                auto descriptor = ProjectManager::CreateTemporaryAsset<LSystemString>();
                                descriptor->Import(m_currentDescriptorPath.m_path);
                                m_currentUsingDescriptor = descriptor;
                            }
                            break;
                        case BehaviourType::SpaceColonization:
                            if (m_currentDescriptorPath.m_isInProjectFolder) {
                                m_currentUsingDescriptor = std::dynamic_pointer_cast<SpaceColonizationParameters>(
                                        ProjectManager::GetOrCreateAsset(m_currentDescriptorPath.m_path));
                            } else {
                                auto descriptor = ProjectManager::CreateTemporaryAsset<SpaceColonizationParameters>();
                                descriptor->Import(m_currentDescriptorPath.m_path);
                                m_currentUsingDescriptor = descriptor;
                            }
                            break;
                        case BehaviourType::TreeGraph:
                            if (m_currentDescriptorPath.m_isInProjectFolder) {
                                m_currentUsingDescriptor = std::dynamic_pointer_cast<TreeGraph>(
                                        ProjectManager::GetOrCreateAsset(m_currentDescriptorPath.m_path));
                            } else {
                                auto descriptor = ProjectManager::CreateTemporaryAsset<TreeGraph>();
                                descriptor->Import(m_currentDescriptorPath.m_path);
                                m_currentUsingDescriptor = descriptor;
                            }
                            break;
                    }
                    m_descriptorPaths.pop_back();
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
                        m_prefix =
                                m_currentDescriptorPath.m_path.filename().stem().string() +
                                "_";
                        break;
                    case BehaviourType::LSystem:
                        m_prefix =
                                m_currentDescriptorPath.m_path.filename().stem().string() +
                                "_";
                        break;
                    case BehaviourType::TreeGraph:
                        m_prefix =
                                m_currentDescriptorPath.m_path.filename().stem().string() +
                                "_";
                        break;
                    case BehaviourType::SpaceColonization:
                        m_prefix =
                                m_currentDescriptorPath.m_path.filename().stem().string() +
                                "_";
                        break;
                }
                m_prefix += std::to_string(GetSeed());
                switch (m_behaviourType) {
                    case BehaviourType::GeneralTree:
                        m_currentGrowingTree = std::dynamic_pointer_cast<GeneralTreeBehaviour>(
                                m_currentInternodeBehaviour)->NewPlant(scene,
                                                                       m_currentUsingDescriptor.Get<GeneralTreeParameters>(),
                                                                       Transform());
                        break;
                    case BehaviourType::TreeGraph:
                        m_currentGrowingTree = m_currentUsingDescriptor.Get<TreeGraph>()->InstantiateTree();
                        break;
                    case BehaviourType::LSystem:
                        m_currentGrowingTree = std::dynamic_pointer_cast<LSystemBehaviour>(
                                m_currentInternodeBehaviour)->NewPlant(scene,
                                                                       m_currentUsingDescriptor.Get<LSystemString>(),
                                                                       Transform());
                        break;
                    case BehaviourType::SpaceColonization:
                        m_currentGrowingTree = std::dynamic_pointer_cast<SpaceColonizationBehaviour>(
                                m_currentInternodeBehaviour)->NewPlant(scene,
                                                                       m_currentUsingDescriptor.Get<SpaceColonizationParameters>(),
                                                                       Transform());
                        break;
                }
                pipelineBehaviour->OnBeforeGrowth(*this);
                if (m_status != AutoTreeGenerationPipelineStatus::BeforeGrowth) {
                    if (!scene->IsEntityValid(m_currentGrowingTree) ||
                        !m_currentInternodeBehaviour->RootCheck(scene, m_currentGrowingTree)) {
                        UNIENGINE_ERROR("No tree created or wrongly created!");
                        m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
                    }
                }
                break;
            case AutoTreeGenerationPipelineStatus::Growth: {
                Application::GetLayer<InternodeLayer>()->Simulate(m_iterations);
                m_status = AutoTreeGenerationPipelineStatus::AfterGrowth;
            }
                break;
            case AutoTreeGenerationPipelineStatus::AfterGrowth:
                pipelineBehaviour->OnAfterGrowth(*this);
                if (m_status != AutoTreeGenerationPipelineStatus::AfterGrowth) {
                    if (scene->IsEntityValid(m_currentGrowingTree))
                        scene->DeleteEntity(m_currentGrowingTree);
                }
                break;
        }
    }
}

static const char *BehaviourTypes[]{"GeneralTree", "LSystem", "SpaceColonization", "TreeGraph"};

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
        ImGui::Text(("Remaining descriptors: " + std::to_string(m_descriptorPaths.size())).c_str());
        ImGui::Text(("Total: " + std::to_string(m_generationAmount) + ", Remaining: " +
                     std::to_string(m_remainingInstanceAmount)).c_str());
        if (ImGui::Button("Force stop")) {
            m_remainingInstanceAmount = 0;
            m_descriptorPaths.clear();
            m_busy = false;
        }
    } else {
        int behaviourType = (int) m_behaviourType;
        if (ImGui::Combo(
                "Plant behaviour type",
                &behaviourType,
                BehaviourTypes,
                IM_ARRAYSIZE(BehaviourTypes))) {
            SetBehaviourType((BehaviourType) behaviourType);
        }
        ImGui::Text(("Loaded descriptors: " + std::to_string(m_descriptorPaths.size())).c_str());
        FileUtils::OpenFolder("Collect descriptors", [&](const std::filesystem::path &path) {
            auto &projectManager = ProjectManager::GetInstance();
            if (ProjectManager::IsInProjectFolder(path)) {
                if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                    for (const auto &entry: std::filesystem::recursive_directory_iterator(path)) {
                        if (!std::filesystem::is_directory(entry.path())) {
                            auto relativePath = ProjectManager::GetPathRelativeToProject(entry.path());
                            switch (m_behaviourType) {
                                case BehaviourType::GeneralTree:
                                    if (entry.path().extension() == ".gtparams") {
                                        m_descriptorPaths.push_back({true, relativePath});
                                        break;
                                    }
                                    break;
                                case BehaviourType::LSystem:
                                    if (entry.path().extension() == ".lstring") {
                                        m_descriptorPaths.push_back({true, relativePath});
                                        break;
                                    }
                                    break;
                                case BehaviourType::SpaceColonization:
                                    if (entry.path().extension() == ".scparams") {
                                        m_descriptorPaths.push_back({true, relativePath});
                                        break;
                                    }
                                    break;
                                case BehaviourType::TreeGraph:
                                    if (entry.path().extension() == ".treegraph") {
                                        m_descriptorPaths.push_back({true, relativePath});
                                        break;
                                    }
                                    break;
                            }
                        }
                    }
                }
            } else {
                if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                    for (const auto &entry: std::filesystem::recursive_directory_iterator(path)) {
                        if (!std::filesystem::is_directory(entry.path())) {
                            switch (m_behaviourType) {
                                case BehaviourType::GeneralTree:
                                    if (entry.path().extension() == ".gtparams") {
                                        m_descriptorPaths.push_back({false, entry.path()});
                                        break;
                                    }
                                    break;
                                case BehaviourType::LSystem:
                                    if (entry.path().extension() == ".lstring") {
                                        m_descriptorPaths.push_back({false, entry.path()});
                                        break;
                                    }
                                    break;
                                case BehaviourType::SpaceColonization:
                                    if (entry.path().extension() == ".scparams") {
                                        m_descriptorPaths.push_back({false, entry.path()});
                                        break;
                                    }
                                    break;
                                case BehaviourType::TreeGraph:
                                    if (entry.path().extension() == ".treegraph") {
                                        m_descriptorPaths.push_back({false, entry.path()});
                                        break;
                                    }
                                    break;
                            }
                        }
                    }
                }
            }

        }, false);
        if (!m_descriptorPaths.empty()) {
            if (ImGui::TreeNodeEx("Loaded descriptors")) {
                for (const auto &i: m_descriptorPaths) {
                    ImGui::Text((i.m_isInProjectFolder ? "T |" : "F |" + i.m_path.string()).c_str());
                }
                ImGui::TreePop();
            }
        }
        if (ImGui::Button("Clear descriptors")) {
            m_descriptorPaths.clear();
        }
        if (m_descriptorPaths.empty()) {
            ImGui::Text("No descriptors!");
        } else if (Application::IsPlaying()) {
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
                    Application::GetLayer<InternodeLayer>()->GetPlantBehaviour<GeneralTreeBehaviour>());
            break;
        case BehaviourType::LSystem:
            m_currentInternodeBehaviour =
                    Application::GetLayer<InternodeLayer>()->GetPlantBehaviour<LSystemBehaviour>();
            break;
        case BehaviourType::SpaceColonization:
            m_currentInternodeBehaviour =
                    Application::GetLayer<InternodeLayer>()->GetPlantBehaviour<SpaceColonizationBehaviour>();
            break;
        case BehaviourType::TreeGraph:
            m_currentInternodeBehaviour =
                    Application::GetLayer<InternodeLayer>()->GetPlantBehaviour<GeneralTreeBehaviour>();
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
    m_descriptorPaths.clear();
}

int AutoTreeGenerationPipeline::GetSeed() const {
    return m_generationAmount - m_remainingInstanceAmount + m_startIndex - 1;
}

void IAutoTreeGenerationPipelineBehaviour::OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) {
}

void IAutoTreeGenerationPipelineBehaviour::OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) {
}

void IAutoTreeGenerationPipelineBehaviour::OnEnd(AutoTreeGenerationPipeline &pipeline) {

}
