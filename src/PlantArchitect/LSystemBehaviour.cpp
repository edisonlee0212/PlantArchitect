//
// Created by lllll on 8/31/2021.
//

#include "LSystemBehaviour.hpp"
#include "InternodeSystem.hpp"
#include "EmptyInternodeResource.hpp"
#include "TransformManager.hpp"
using namespace PlantArchitect;

void LSystemBehaviour::OnInspect() {
    RecycleButton();

    static LSystemParameters parameters;
    parameters.OnInspect();
    FileUtils::OpenFile("Load L-String", "L-String", {".txt"}, [&](const std::filesystem::path &path) {
        auto string = FileUtils::LoadFileAsString(path);
        std::vector<LSystemCommand> commands;
        ParseLString(string, commands);
        if (!commands.empty()) {
            Entity entity = FormPlant(commands, parameters);
        }
    });

    static float resolution = 0.02;
    static float subdivision = 4.0;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate branch mesh")) {
        GenerateBranchSkinnedMeshes(m_internodesQuery, subdivision, resolution);
    }

}

void LSystemBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled L-System Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("L-System Internode", InternodeInfo(),
                                                 LSystemTag(), LSystemParameters(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(LSystemTag());
}


void LSystemBehaviour::ParseLString(const std::string &string, std::vector<LSystemCommand> &commands) {
    std::istringstream iss(string);
    std::string line;
    int stackCheck = 0;
    while (std::getline(iss, line)) {
        LSystemCommand command;
        switch (line[0]) {
            case 'F': {
                command.m_type = LSystemCommandType::Forward;
            }
                break;
            case '+': {
                command.m_type = LSystemCommandType::TurnLeft;
            }
                break;
            case '-': {
                command.m_type = LSystemCommandType::TurnRight;
            }
                break;
            case '^': {
                command.m_type = LSystemCommandType::PitchUp;
            }
                break;
            case '&': {
                command.m_type = LSystemCommandType::PitchDown;
            }
                break;
            case '\\': {
                command.m_type = LSystemCommandType::RollLeft;
            }
                break;
            case '/': {
                command.m_type = LSystemCommandType::RollRight;
            }
                break;
            case '[': {
                command.m_type = LSystemCommandType::Push;
                stackCheck++;
            }
                break;
            case ']': {
                command.m_type = LSystemCommandType::Pop;
                stackCheck--;
            }
                break;
        }
        if (command.m_type != LSystemCommandType::Push && command.m_type != LSystemCommandType::Pop) {
            command.m_value = std::stof(line.substr(2));
            if (command.m_type == LSystemCommandType::Forward && command.m_value > 2.0f) {
                command.m_value = 3.0f;
            }
        }
        commands.push_back(command);
    }
    if (stackCheck != 0) {
        UNIENGINE_ERROR("Stack check failed! Something wrong with the string!");
        commands.clear();
    }
}

Entity LSystemBehaviour::FormPlant(std::vector<LSystemCommand> &commands, const LSystemParameters &parameters) {
    if (commands.empty()) return Entity();
    Entity currentNode = Retrieve();
    Entity root = currentNode;
    currentNode.SetDataComponent(parameters);
    InternodeInfo newInfo;
    newInfo.m_length = 0;
    newInfo.m_thickness = 0.1f;
    currentNode.SetDataComponent(newInfo);
    for (const auto &command: commands) {
        Transform transform = currentNode.GetDataComponent<Transform>();
        InternodeInfo internodeInfo = currentNode.GetDataComponent<InternodeInfo>();
        switch (command.m_type) {
            case LSystemCommandType::Forward: {
                internodeInfo.m_length += command.m_value;
                currentNode.SetDataComponent(internodeInfo);
            }
                continue;
            case LSystemCommandType::PitchUp: {
                auto currentEulerRotation = transform.GetEulerRotation();
                currentEulerRotation.x += command.m_value;
                transform.SetEulerRotation(currentEulerRotation);
            }
                break;
            case LSystemCommandType::PitchDown: {
                auto currentEulerRotation = transform.GetEulerRotation();
                currentEulerRotation.x -= command.m_value;
                transform.SetEulerRotation(currentEulerRotation);
            }
                break;
            case LSystemCommandType::TurnLeft: {
                auto currentEulerRotation = transform.GetEulerRotation();
                currentEulerRotation.y += command.m_value;
                transform.SetEulerRotation(currentEulerRotation);
            }
                break;
            case LSystemCommandType::TurnRight: {
                auto currentEulerRotation = transform.GetEulerRotation();
                currentEulerRotation.y -= command.m_value;
                transform.SetEulerRotation(currentEulerRotation);
            }
                break;
            case LSystemCommandType::RollLeft: {
                auto currentEulerRotation = transform.GetEulerRotation();
                currentEulerRotation.z += command.m_value;
                transform.SetEulerRotation(currentEulerRotation);
            }
                break;
            case LSystemCommandType::RollRight: {
                auto currentEulerRotation = transform.GetEulerRotation();
                currentEulerRotation.z -= command.m_value;
                transform.SetEulerRotation(currentEulerRotation);
            }
                break;
            case LSystemCommandType::Push: {
                currentNode = Retrieve(currentNode);
                currentNode.SetDataComponent(parameters);
                InternodeInfo newInfo;
                newInfo.m_length = 0;
                newInfo.m_thickness = 0.1f;
                currentNode.SetDataComponent(newInfo);
                continue;
            }
            case LSystemCommandType::Pop: {
                currentNode = currentNode.GetParent();
                continue;
            }
        }
        currentNode.SetDataComponent(transform);
        if(currentNode == root){
            GlobalTransform globalTransform;
            globalTransform.m_value = transform.m_value;
            currentNode.SetDataComponent(globalTransform);
        }
    }
    TransformManager::CalculateTransformGraphForDescendents(root);
    TreeGraphWalkerRootToEnd(root, root, [](Entity parent, Entity child) {
        auto parentGlobalTransform = parent.GetDataComponent<GlobalTransform>();
        auto parentPosition = parentGlobalTransform.GetPosition();
        auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
        auto childPosition = parentPosition + parentInternodeInfo.m_length * (parentGlobalTransform.GetRotation() * glm::vec3(0, 0, -1));
        auto childGlobalTransform = child.GetDataComponent<GlobalTransform>();
        childGlobalTransform.SetPosition(childPosition);
        child.SetDataComponent(childGlobalTransform);
    });

    TreeGraphWalkerEndToRoot(root, root, [&](Entity parent) {
        float thicknessCollection = 0.0f;
        auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
        auto parameters = parent.GetDataComponent<LSystemParameters>();
        parent.ForEachChild([&](Entity child) {
            if (!InternodeCheck(child)) return;
            auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            thicknessCollection += glm::pow(childInternodeInfo.m_thickness,
                                            1.0f / parameters.m_thicknessFactor);
        });
        parentInternodeInfo.m_thickness = glm::pow(thicknessCollection, parameters.m_thicknessFactor);
        parent.SetDataComponent(parentInternodeInfo);
    }, [](Entity endNode) {
        auto internodeInfo = endNode.GetDataComponent<InternodeInfo>();
        auto parameters = endNode.GetDataComponent<LSystemParameters>();
        internodeInfo.m_thickness = parameters.m_endNodeThickness;
        endNode.SetDataComponent(internodeInfo);
    });
    return root;
}

bool LSystemBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<LSystemTag>();
}

Entity LSystemBehaviour::Retrieve() {
    return RetrieveHelper<EmptyInternodeResource>();
}

Entity LSystemBehaviour::Retrieve(const Entity &parent) {
    return RetrieveHelper<EmptyInternodeResource>(parent);
}


void LSystemParameters::OnInspect() {
    ImGui::DragFloat("Internode Length", &m_internodeLength);
    ImGui::DragFloat("Thickness Factor", &m_thicknessFactor);
    ImGui::DragFloat("End node thickness", &m_endNodeThickness);
}
