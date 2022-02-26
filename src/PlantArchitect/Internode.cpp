//
// Created by lllll on 8/27/2021.
//

#include "Internode.hpp"
#include "InternodeLayer.hpp"
#include "LSystemBehaviour.hpp"
#include "AssetManager.hpp"
#include "InternodeFoliage.hpp"

using namespace PlantArchitect;

void BranchPhysicsParameters::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_density" << YAML::Value << m_density;
    out << YAML::Key << "m_linearDamping" << YAML::Value << m_linearDamping;
    out << YAML::Key << "m_angularDamping" << YAML::Value << m_angularDamping;
    out << YAML::Key << "m_positionSolverIteration" << YAML::Value << m_positionSolverIteration;
    out << YAML::Key << "m_velocitySolverIteration" << YAML::Value << m_velocitySolverIteration;
    out << YAML::Key << "m_jointDriveStiffnessFactor" << YAML::Value << m_jointDriveStiffnessFactor;
    out << YAML::Key << "m_jointDriveStiffnessThicknessFactor" << YAML::Value << m_jointDriveStiffnessThicknessFactor;
    out << YAML::Key << "m_jointDriveDampingFactor" << YAML::Value << m_jointDriveDampingFactor;
    out << YAML::Key << "m_jointDriveDampingThicknessFactor" << YAML::Value << m_jointDriveDampingThicknessFactor;
    out << YAML::Key << "m_enableAccelerationForDrive" << YAML::Value << m_enableAccelerationForDrive;
}

void BranchPhysicsParameters::Deserialize(const YAML::Node &in) {
    if (in["m_density"]) m_density = in["m_density"].as<float>();
    if (in["m_linearDamping"]) m_linearDamping = in["m_linearDamping"].as<float>();
    if (in["m_angularDamping"]) m_angularDamping = in["m_angularDamping"].as<float>();
    if (in["m_positionSolverIteration"]) m_positionSolverIteration = in["m_positionSolverIteration"].as<int>();
    if (in["m_velocitySolverIteration"]) m_velocitySolverIteration = in["m_velocitySolverIteration"].as<int>();
    if (in["m_jointDriveStiffnessFactor"]) m_jointDriveStiffnessFactor = in["m_jointDriveStiffnessFactor"].as<float>();
    if (in["m_jointDriveStiffnessThicknessFactor"]) m_jointDriveStiffnessThicknessFactor = in["m_jointDriveStiffnessThicknessFactor"].as<float>();
    if (in["m_jointDriveDampingFactor"]) m_jointDriveDampingFactor = in["m_jointDriveDampingFactor"].as<float>();
    if (in["m_jointDriveDampingThicknessFactor"]) m_jointDriveDampingThicknessFactor = in["m_jointDriveDampingThicknessFactor"].as<float>();
    if (in["m_enableAccelerationForDrive"]) m_enableAccelerationForDrive = in["m_enableAccelerationForDrive"].as<bool>();
}

void BranchPhysicsParameters::OnInspect() {
    if (ImGui::TreeNodeEx("Physics")) {
        ImGui::DragFloat("Internode Density", &m_density, 0.1f, 0.01f, 1000.0f);
        ImGui::DragFloat2("RigidBody Damping", &m_linearDamping, 0.1f, 0.01f,
                          1000.0f);
        ImGui::DragFloat2("Drive Stiffness", &m_jointDriveStiffnessFactor, 0.1f,
                          0.01f, 1000000.0f);
        ImGui::DragFloat2("Drive Damping", &m_jointDriveDampingFactor, 0.1f, 0.01f,
                          1000000.0f);
        ImGui::Checkbox("Use acceleration", &m_enableAccelerationForDrive);

        int pi = m_positionSolverIteration;
        int vi = m_velocitySolverIteration;
        if (ImGui::DragInt("Velocity solver iteration", &vi, 1, 1, 100)) {
            m_velocitySolverIteration = vi;
        }
        if (ImGui::DragInt("Position solver iteration", &pi, 1, 1, 100)) {
            m_positionSolverIteration = pi;
        }
        ImGui::TreePop();
    }
}

void Internode::OnCreate() {
    if (m_resource) m_resource->Reset();
    m_normalDir = glm::vec3(0, 0, 1);
    m_step = 4;
    m_foliageMatrices.clear();
    m_rings.clear();
    m_apicalBud.m_status = BudStatus::Sleeping;
    m_lateralBuds.clear();
    m_fromApicalBud = true;
    GetOwner().SetDataComponent(InternodeInfo());
}


void Internode::DownStreamResource(float deltaTime) {
    if (!m_resource) return;
    auto owner = GetOwner();
    m_resource->DownStream(deltaTime, owner, owner.GetParent());
}

void Internode::UpStreamResource(float deltaTime) {
    if (!m_resource) return;
    auto owner = GetOwner();
    auto children = owner.GetChildren();
    for (const auto &child: children) {
        m_resource->UpStream(deltaTime, owner, child);
    }
}

void Internode::CollectResource(float deltaTime) {
    if (!m_resource) return;
    m_resource->Collect(deltaTime, GetOwner());
}

void Internode::CollectInternodesHelper(const Entity &target, std::vector<Entity> &results) {
    if (target.IsValid() && target.HasDataComponent<InternodeInfo>() && target.HasPrivateComponent<Internode>()) {
        results.push_back(target);
        target.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            CollectInternodesHelper(child, results);
        });
    }
}

void Internode::CollectInternodes(std::vector<Entity> &results) {
    CollectInternodesHelper(GetOwner(), results);
}

void Internode::ExportLString(const std::shared_ptr<LString> &lString) {
    int index = 0;
    ExportLSystemCommandsHelper(index, GetOwner(), lString->commands);
}

void Internode::ExportLSystemCommandsHelper(int &index, const Entity &target, std::vector<LSystemCommand> &commands) {
    if (!target.IsValid() || !target.HasDataComponent<InternodeInfo>()) return;
    auto internodeInfo = target.GetDataComponent<InternodeInfo>();
    auto internodeStatstics = target.GetDataComponent<InternodeStatistics>();
    auto transform = target.GetDataComponent<Transform>();
    auto eulerRotation = transform.GetEulerRotation();
    if (eulerRotation.x > 0) {
        commands.push_back({LSystemCommandType::PitchUp, eulerRotation.x});
        index++;
    } else if (eulerRotation.x < 0) {
        commands.push_back({LSystemCommandType::PitchDown, -eulerRotation.x});
        index++;
    }
    if (eulerRotation.y > 0) {
        commands.push_back({LSystemCommandType::TurnLeft, eulerRotation.y});
        index++;
    } else if (eulerRotation.y < 0) {
        commands.push_back({LSystemCommandType::TurnRight, -eulerRotation.y});
        index++;
    }
    if (eulerRotation.z > 0) {
        commands.push_back({LSystemCommandType::RollLeft, eulerRotation.z});
        index++;
    } else if (eulerRotation.z < 0) {
        commands.push_back({LSystemCommandType::RollRight, -eulerRotation.z});
        index++;
    }
    commands.push_back({LSystemCommandType::Forward, internodeInfo.m_length});
    internodeStatstics.m_lSystemStringIndex = index;
    target.SetDataComponent(internodeInfo);
    index++;

    target.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (!child.IsValid() || !child.HasDataComponent<InternodeInfo>()) return;
        commands.push_back({LSystemCommandType::Push, 0.0f});
        index++;
        ExportLSystemCommandsHelper(index, child, commands);
        commands.push_back({LSystemCommandType::Pop, 0.0f});
        index++;
    });
}

void Internode::OnInspect() {
    if (ImGui::Button("Generate L-String")) {
        auto lString = AssetManager::CreateAsset<LString>();
        AssetManager::Share(lString);
        ExportLString(lString);
    }
    if (ImGui::Button("Calculate L-String Indices")) {
        int index = 0;
        std::vector<LSystemCommand> commands;
        ExportLSystemCommandsHelper(index, GetOwner(), commands);
    }
    m_foliage.Get<InternodeFoliage>()->OnInspect();

    if (ImGui::TreeNodeEx("Buds")) {
        ImGui::Text("Apical bud:");
        m_apicalBud.OnInspect();
        ImGui::Text("Lateral bud:");
        for (int i = 0; i < m_lateralBuds.size(); i++) {
            if (ImGui::TreeNodeEx(("Bud " + std::to_string(i)).c_str())) {
                m_lateralBuds[i].OnInspect();
                ImGui::TreePop();
            }
        }

        ImGui::TreePop();
    }
    m_branchPhysicsParameters.OnInspect();
}

void Internode::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_foliage);
}

void Internode::Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) {
    m_currentRoot.Relink(map);
}

void Internode::PostCloneAction(const std::shared_ptr<IPrivateComponent> &target) {

}

void Internode::Serialize(YAML::Emitter &out) {
    m_currentRoot.Save("m_currentRoot", out);
    m_foliage.Save("m_foliage", out);
    m_apicalBud.Save("m_apicalBud", out);
    SaveList("m_lateralBuds", m_lateralBuds, out);

    out << YAML::Key << "m_fromApicalBud" << YAML::Value << m_fromApicalBud;


    out << YAML::Key << "m_branchPhysicsParameters" << YAML::Value << YAML::BeginMap;
    m_branchPhysicsParameters.Serialize(out);
    out << YAML::EndMap;

}

void Internode::Deserialize(const YAML::Node &in) {
    m_currentRoot.Load("m_currentRoot", in);
    m_foliage.Load("m_foliage", in);
    m_apicalBud.Load("m_apicalBud", in);
    LoadList("m_lateralBuds", m_lateralBuds, in);
    m_fromApicalBud = in["m_fromApicalBud"].as<bool>();

    if (in["m_branchPhysicsParameters"]) m_branchPhysicsParameters.Deserialize(in["m_branchPhysicsParameters"]);
}

void Bud::OnInspect() {
    ImGui::Text("Flush prob: %.3f", m_flushProbability);
    switch (m_status) {
        case BudStatus::Sleeping:
            ImGui::Text("Status: Sleeping");
            break;
        case BudStatus::Flushing:
            ImGui::Text("Status: Flushing");
            break;
        case BudStatus::Flushed:
            ImGui::Text("Status: Flushed");
            break;
        case BudStatus::Died:
            ImGui::Text("Status: Died");
            break;
    }
}

void Bud::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_flushProbability" << YAML::Value << m_flushProbability;
    out << YAML::Key << "m_status" << YAML::Value << (unsigned) m_status;
}

void Bud::Deserialize(const YAML::Node &in) {
    m_status = (BudStatus) in["m_status"].as<unsigned>();
    m_flushProbability = in["m_flushProbability"].as<float>();
}

void Bud::Save(const std::string &name, YAML::Emitter &out) {
    out << YAML::Key << name << YAML::BeginMap;
    Serialize(out);
    out << YAML::EndMap;
}

void Bud::Load(const std::string &name, const YAML::Node &in) {
    if (in[name]) Deserialize(in[name]);
}