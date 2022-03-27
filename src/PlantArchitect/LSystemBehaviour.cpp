//
// Created by lllll on 8/31/2021.
//

#include "LSystemBehaviour.hpp"
#include "PlantLayer.hpp"
#include "EmptyInternodeResource.hpp"
#include "TransformLayer.hpp"
#include "DefaultInternodePhyllotaxis.hpp"

using namespace PlantArchitect;

void LSystemBehaviour::OnInspect() {
}

LSystemBehaviour::LSystemBehaviour() {
    m_typeName = "LSystemBehaviour";
    m_internodeArchetype =
            Entities::CreateEntityArchetype("L-System Internode", InternodeInfo(), InternodeStatistics(),
                                            LSystemTag(),
                                            InternodeColor(), InternodeCylinder(), InternodeCylinderWidth(),
                                            InternodePointer());
    m_internodesQuery = Entities::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo(), LSystemTag());

    m_rootArchetype =
            Entities::CreateEntityArchetype("L-System Root", RootInfo(), LSystemTag());
    m_rootsQuery = Entities::CreateEntityQuery();
    m_rootsQuery.SetAllFilters(RootInfo(), LSystemTag());

    m_branchArchetype =
            Entities::CreateEntityArchetype("L-System Branch", BranchInfo(),
                                            LSystemTag(),
                                            BranchColor(), BranchCylinder(), BranchCylinderWidth());
    m_branchesQuery = Entities::CreateEntityQuery();
    m_branchesQuery.SetAllFilters(BranchInfo(), LSystemTag());
}


Entity LSystemBehaviour::NewPlant(AssetRef descriptor, const Transform &transform) {
    auto lString = descriptor.Get<LSystemString>();
    auto &commands = lString->m_commands;
    if (commands.empty()) return {};
    LSystemState currentState;
    //The stack that records current state when a push happens and restore previous state when pop.
    std::vector<LSystemState> stateStack;
    //The stack that keeps track of current internode (entity) during push/pop
    std::vector<Entity> entityStack;
    int index = 1;
    Entity root;
    Entity rootInternode;
    Entity internode;
    Entity rootBranch;
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
                        //We need to create a child node with this function. Here CreateInternode(Entity) will instantiate a new internode for you and set it as a child of current internode.
                        internode = CreateInternode(internode);
                    } else {
                        //Calculate the local rotation as quaternion from euler angles.
                        newInfo.m_localRotation = glm::quat(currentState.m_eulerRotation);
                        //If this is the first push in the string, we create the root internode.
                        //The node creation is handled by the CreateInternode() function. The internode creation is performed in a factory pattern.
                        root = CreateRoot(descriptor, rootInternode, rootBranch);
                        root.GetOrSetPrivateComponent<Root>().lock()->m_foliagePhyllotaxis = AssetManager::CreateAsset<DefaultInternodePhyllotaxis>();
                        internode = rootInternode;
                        rootExists = true;
                    }
                } else {
                    //Calculate the local rotation as quaternion from euler angles.
                    newInfo.m_localRotation = glm::quat(currentState.m_eulerRotation);
                    //We need to create a child node with this function. Here CreateInternode(Entity) will instantiate a new internode for you and set it as a child of current internode.
                    internode = CreateInternode(internode);
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
    rootTransform.SetRotation(rootInternode.GetDataComponent<InternodeInfo>().m_localRotation);
    rootInternode.SetDataComponent(rootTransform);

    //Since we only stored the rotation data into internode info without applying it to the local transformation matrix of the internode, we do it here.
    InternodeGraphWalkerRootToEnd(rootInternode, [](Entity parent, Entity child) {
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
    auto params = descriptor.Get<LSystemString>();
    //Calculate other properties like thickness after the structure of the tree is ready.
    InternodeGraphWalkerEndToRoot(rootInternode, [&](Entity parent) {
        float thicknessCollection = 0.0f;
        auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
        parent.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!InternodeCheck(child)) return;
            auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            thicknessCollection += glm::pow(childInternodeInfo.m_thickness,
                                            1.0f / params->m_thicknessFactor);
        });
        parentInternodeInfo.m_thickness = glm::pow(thicknessCollection, params->m_thicknessFactor);
        parent.SetDataComponent(parentInternodeInfo);
    }, [&](Entity endNode) {
        auto internodeInfo = endNode.GetDataComponent<InternodeInfo>();
        internodeInfo.m_thickness = params->m_endNodeThickness;
        endNode.SetDataComponent(internodeInfo);
    });

    Application::GetLayer<PlantLayer>()->CalculateStatistics();
    UpdateBranches();
    return root;
}

bool LSystemBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<LSystemTag>();
}

Entity LSystemBehaviour::CreateInternode(const Entity &parent) {
    return CreateInternodeHelper<EmptyInternodeResource>(parent);
}

bool LSystemBehaviour::InternalRootCheck(const Entity &target) {
    return target.HasDataComponent<LSystemTag>();
}

bool LSystemBehaviour::InternalBranchCheck(const Entity &target) {
    return target.HasDataComponent<LSystemTag>();
}

Entity LSystemBehaviour::CreateRoot(AssetRef descriptor, Entity &rootInternode, Entity &rootBranch) {
    return CreateRootHelper<EmptyInternodeResource>(descriptor, rootInternode, rootBranch);
}

Entity LSystemBehaviour::CreateBranch(const Entity &parent, const Entity &internode) {
    return CreateBranchHelper(parent, internode);
}

bool LSystemString::LoadInternal(const std::filesystem::path &path) {
    if (path.extension().string() == ".lstring") {
        auto string = FileUtils::LoadFileAsString(path);
        ParseLString(string);
        return true;
    }
    return false;
}

bool LSystemString::SaveInternal(const std::filesystem::path &path) {
    if (path.extension().string() == ".lstring") {
        std::ofstream of;
        of.open(path.c_str(),
                std::ofstream::out | std::ofstream::trunc);
        if (of.is_open()) {
            std::string output;
            for (const auto &command: m_commands) {
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

void LSystemString::ParseLString(const std::string &string) {
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
        m_commands.push_back(command);
    }
    if (stackCheck != 0) {
        UNIENGINE_ERROR("Stack check failed! Something wrong with the string!");
        m_commands.clear();
    }
}

void LSystemString::OnInspect() {
    IPlantDescriptor::OnInspect();
    ImGui::Text(("Command Size: " + std::to_string(m_commands.size())).c_str());
    ImGui::DragFloat("Internode Length", &m_internodeLength);
    ImGui::DragFloat("Thickness Factor", &m_thicknessFactor);
    ImGui::DragFloat("End node thickness", &m_endNodeThickness);
}

Entity LSystemString::InstantiateTree() {
    return Application::GetLayer<PlantLayer>()->GetPlantBehaviour<LSystemBehaviour>()->NewPlant(
            AssetManager::Get<LSystemString>(GetHandle()), Transform());
}
