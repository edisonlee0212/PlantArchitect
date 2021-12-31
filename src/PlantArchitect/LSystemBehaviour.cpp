//
// Created by lllll on 8/31/2021.
//

#include "LSystemBehaviour.hpp"
#include "InternodeLayer.hpp"
#include "EmptyInternodeResource.hpp"
#include "TransformLayer.hpp"
#include "InternodeFoliage.hpp"

using namespace PlantArchitect;

void LSystemBehaviour::OnInspect() {
    RecycleButton();
    static float resolution = 0.02;
    static float subdivision = 4.0;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate meshes")) {
        GenerateSkinnedMeshes(subdivision, resolution);
    }

}

void LSystemBehaviour::OnCreate() {
    m_internodeArchetype =
            Entities::CreateEntityArchetype("L-System Internode", InternodeInfo(), InternodeStatistics(),
                                                 LSystemTag(), LSystemParameters(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer(), BranchPhysicsParameters());
    m_internodesQuery = Entities::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(LSystemTag());
}


Entity LSystemBehaviour::FormPlant(const std::shared_ptr<LString> &lString, const LSystemParameters &parameters) {
    auto &commands = lString->commands;
    if (commands.empty()) return {};
    LSystemState currentState;
    //The stack that records current state when a push happens and restore previous state when pop.
    std::vector<LSystemState> stateStack;
    //The stack that keeps track of current internode (entity) during push/pop
    std::vector<Entity> entityStack;
    int index = 1;
    Entity root;
    Entity internode;
    bool rootExists = false;
    for (const auto &command: commands) {
        switch (command.m_type) {
            case LSystemCommandType::Forward: {
                InternodeInfo newInfo;
                InternodeStatistics newStat;
                newStat.m_lSystemStringIndex = index;
                if (internode.IsNull()) {
                    if (rootExists) {
                        UNIENGINE_WARNING("Root exists!");
                        //Calculate the local rotation as quaternion from euler angles.
                        newInfo.m_localRotation = glm::quat(currentState.m_eulerRotation);
                        //We need to create a child node with this function. Here Retrieve(Entity) will instantiate a new internode for you and set it as a child of current internode.
                        internode = Retrieve(root);
                        //Apply the parameter to this internode, this is necessary everytime you create a new internode.
                        internode.SetDataComponent(parameters);
                    }else {
                        //Calculate the local rotation as quaternion from euler angles.
                        newInfo.m_localRotation = glm::quat(currentState.m_eulerRotation);
                        //If this is the first push in the string, we create the root internode.
                        //The node creation is handled by the Retrieve() function. The internode creation is performed in a factory pattern.
                        root = internode = Retrieve();
                        //Apply the parameter to this internode, this is necessary everytime you create a new internode.
                        internode.SetDataComponent(parameters);
                        rootExists = true;
                    }
                } else {
                    //Calculate the local rotation as quaternion from euler angles.
                    newInfo.m_localRotation = glm::quat(currentState.m_eulerRotation);
                    //We need to create a child node with this function. Here Retrieve(Entity) will instantiate a new internode for you and set it as a child of current internode.
                    internode = Retrieve(internode);
                    //Apply the parameter to this internode, this is necessary everytime you create a new internode.
                    internode.SetDataComponent(parameters);
                }
                newInfo.m_length = command.m_value;
                newInfo.m_thickness = 0.2f;
                internode.SetDataComponent(newInfo);
                internode.SetDataComponent(newStat);
            }
                break;
            case LSystemCommandType::PitchUp: {
                //Update current state
                currentState.m_eulerRotation.x += command.m_value;
            }
                break;
            case LSystemCommandType::PitchDown: {
                //Update current state
                currentState.m_eulerRotation.x -= command.m_value;
            }
                break;
            case LSystemCommandType::TurnLeft: {
                //Update current state
                currentState.m_eulerRotation.y += command.m_value;
            }
                break;
            case LSystemCommandType::TurnRight: {
                //Update current state
                currentState.m_eulerRotation.y -= command.m_value;
            }
                break;
            case LSystemCommandType::RollLeft: {
                //Update current state
                currentState.m_eulerRotation.z += command.m_value;
            }
                break;
            case LSystemCommandType::RollRight: {
                //Update current state
                currentState.m_eulerRotation.z -= command.m_value;
            }
                break;
            case LSystemCommandType::Push: {
                //Update stack
                stateStack.push_back(currentState);
                currentState.m_eulerRotation = glm::vec3(0.0f);
                entityStack.push_back(internode);
                break;
            }
            case LSystemCommandType::Pop: {
                //Update stack
                currentState = stateStack.back();
                stateStack.pop_back();
                internode = entityStack.back();
                entityStack.pop_back();
                break;
            }
        }
        index++;
    }

    Transform rootTransform;
    rootTransform.SetRotation(root.GetDataComponent<InternodeInfo>().m_localRotation);
    root.SetDataComponent(rootTransform);

    //Since we only stored the rotation data into internode info without applying it to the local transformation matrix of the internode, we do it here.
    TreeGraphWalkerRootToEnd(root, root, [](Entity parent, Entity child) {
        auto parentGlobalTransform = parent.GetDataComponent<GlobalTransform>();
        auto parentPosition = parentGlobalTransform.GetPosition();
        auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
        auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
        auto parentRotation = parentGlobalTransform.GetRotation();
        auto childPosition = parentInternodeInfo.m_length * (parentRotation * glm::vec3(0, 0, -1));
        auto childRotation = childInternodeInfo.m_localRotation;
        auto childTransform = child.GetDataComponent<Transform>();
        childTransform.SetRotation(childRotation);
        childTransform.SetPosition(childPosition);
        child.SetDataComponent(childTransform);
    });

    //This is another way to do above procedures but in multi-threaded way.
    /*
    Entities::ForEach<GlobalTransform, Transform, InternodeInfo>
            (Entities::GetCurrentScene(),
             JobManager::PrimaryWorkers(), m_internodesQuery,
             [](int i, Entity entity,
                GlobalTransform &globalTransform,
                Transform &transform,
                InternodeInfo &internodeInfo) {
                 auto parent = entity.GetParent();
                 if(parent.IsNull()) return;
                 auto parentGlobalTransform = parent.GetDataComponent<GlobalTransform>();
                 auto parentPosition = parentGlobalTransform.GetPosition();
                 auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                 auto parentRotation = parentGlobalTransform.GetRotation();
                 auto childPosition =
                         parentInternodeInfo.m_length *
                         (parentRotation *
                          glm::vec3(0, 0, -1));
                 auto childRotation = internodeInfo.m_localRotation;
                 transform.SetRotation(childRotation);
                 transform.SetPosition(childPosition);
             }
            );
    */

    //After you setup the local transformation matrix, you call this function to calculate the world(global) transformation matrix from the root of the plant.
    Application::GetLayer<TransformLayer>()->CalculateTransformGraphForDescendents(Entities::GetCurrentScene(),
                                                                                   root);

    //Calculate other properties like thickness after the structure of the tree is ready.
    TreeGraphWalkerEndToRoot(root, root, [&](Entity parent) {
        float thicknessCollection = 0.0f;
        auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
        auto parameters = parent.GetDataComponent<LSystemParameters>();
        parent.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
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

    Application::GetLayer<InternodeLayer>()->CalculateStatistics();
    return root;
}

bool LSystemBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<LSystemTag>();
}

Entity LSystemBehaviour::Retrieve() {
    auto retVal = RetrieveHelper<EmptyInternodeResource>();
    retVal.SetDataComponent(BranchPhysicsParameters());
    return retVal;
}

Entity LSystemBehaviour::Retrieve(const Entity &parent) {
    auto retVal = RetrieveHelper<EmptyInternodeResource>(parent);
    retVal.SetDataComponent(parent.GetDataComponent<BranchPhysicsParameters>());
    return retVal;
}


void LSystemParameters::OnInspect() {
    ImGui::DragFloat("Internode Length", &m_internodeLength);
    ImGui::DragFloat("Thickness Factor", &m_thicknessFactor);
    ImGui::DragFloat("End node thickness", &m_endNodeThickness);
}

bool LString::LoadInternal(const std::filesystem::path &path) {
    if (path.extension().string() == ".lstring") {
        auto string = FileUtils::LoadFileAsString(path);
        ParseLString(string);
        return true;
    }
    return false;
}

bool LString::SaveInternal(const std::filesystem::path &path) {
    if (path.extension().string() == ".lstring") {
        std::ofstream of;
        of.open(path.c_str(),
                std::ofstream::out | std::ofstream::trunc);
        if (of.is_open()) {
            std::string output;
            for (const auto &command: commands) {
                switch (command.m_type) {
                    case LSystemCommandType::Forward: {
                        output += "F(";
                        output += std::to_string(command.m_value);
                        output += ")";
                    }
                        break;
                    case LSystemCommandType::PitchUp: {
                        output += "^(";
                        output += std::to_string(command.m_value);
                        output += ")";
                    }
                        break;
                    case LSystemCommandType::PitchDown: {
                        output += "&(";
                        output += std::to_string(command.m_value);
                        output += ")";
                    }
                        break;
                    case LSystemCommandType::TurnLeft: {
                        output += "+(";
                        output += std::to_string(command.m_value);
                        output += ")";
                    }
                        break;
                    case LSystemCommandType::TurnRight: {
                        output += "-(";
                        output += std::to_string(command.m_value);
                        output += ")";
                    }
                        break;
                    case LSystemCommandType::RollLeft: {
                        output += "\\(";
                        output += std::to_string(command.m_value);
                        output += ")";
                    }
                        break;
                    case LSystemCommandType::RollRight: {
                        output += "/(";
                        output += std::to_string(command.m_value);
                        output += ")";
                    }
                        break;
                    case LSystemCommandType::Push: {
                        output += "[";
                    }
                        break;
                    case LSystemCommandType::Pop: {
                        output += "]";
                    }
                        break;
                }
                output += "\n";
            }
            of.write(output.c_str(), output.size());
            of.flush();
        }
        return true;
    }
    return false;
}

void LString::ParseLString(const std::string &string) {
    std::istringstream iss(string);
    std::string line;
    int stackCheck = 0;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
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
        if (command.m_type == LSystemCommandType::Unknown) continue;
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

void LString::OnInspect() {
    if (ImGui::Button("Instantiate")) {
        auto parameters = LSystemParameters();
        Application::GetLayer<InternodeLayer>()->GetInternodeBehaviour<LSystemBehaviour>()->FormPlant(
                AssetManager::Get<LString>(GetHandle()), parameters);
    }
}
