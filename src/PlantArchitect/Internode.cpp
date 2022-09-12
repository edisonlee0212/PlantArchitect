//
// Created by lllll on 8/27/2021.
//

#include "InternodeModel/Internode.hpp"
#include "InternodeLayer.hpp"
#include "LSystemBehaviour.hpp"
#include "ProjectManager.hpp"

using namespace PlantArchitect;

void Internode::OnCreate() {
    if (m_resource) m_resource->Reset();
    m_normalDir = glm::vec3(0, 0, 1);
    m_step = 4;
    m_foliageMatrices.clear();
    m_rings.clear();
    m_apicalBud.m_status = BudStatus::Sleeping;
    m_lateralBuds.clear();
    m_fromApicalBud = true;
    m_currentRoot.Clear();
    m_resource.reset();

}


void Internode::DownStreamResource(float deltaTime) {
    if (!m_resource) return;
    auto owner = GetOwner();
    auto scene = GetScene();
    m_resource->DownStream(deltaTime, owner, scene->GetParent(owner));
}

void Internode::UpStreamResource(float deltaTime) {
    if (!m_resource) return;
    auto owner = GetOwner();
    auto scene = GetScene();
    auto children = scene->GetChildren(owner);
    for (const auto &child: children) {
        m_resource->UpStream(deltaTime, owner, child);
    }
}

void Internode::CollectResource(float deltaTime) {
    if (!m_resource) return;
    m_resource->Collect(deltaTime, GetOwner());
}

void Internode::CollectInternodesHelper(const Entity &target, std::vector<Entity> &results) {
    auto scene = GetScene();
    if (scene->IsEntityValid(target) && scene->HasDataComponent<InternodeInfo>(target) && scene->HasPrivateComponent<Internode>(target)) {
        results.push_back(target);
        scene->ForEachChild(target, [&](Entity child) {
            CollectInternodesHelper(child, results);
        });
    }
}

void Internode::CollectInternodes(std::vector<Entity> &results) {
    CollectInternodesHelper(GetOwner(), results);
}

void Internode::ExportLString(const std::shared_ptr<LSystemString> &lString) {
    int index = 0;
    ExportLSystemCommandsHelper(index, GetOwner(), lString->m_commands);
}

void Internode::ExportLSystemCommandsHelper(int &index, const Entity &target, std::vector<LSystemCommand> &commands) {
    auto scene = GetScene();
    if (!scene->IsEntityValid(target) || !scene->HasDataComponent<InternodeInfo>(target)) return;
    auto internodeInfo = scene->GetDataComponent<InternodeInfo>(target);
    auto internodeStatstics = scene->GetDataComponent<InternodeStatistics>(target);
    auto transform = scene->GetDataComponent<Transform>(target);
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
    scene->SetDataComponent(target, internodeInfo);
    index++;
    auto children = scene->GetChildren(target);
    for(int i = 1; i < children.size(); i++){
        auto& child = children[i];
        if (!scene->IsEntityValid(child) || !scene->HasDataComponent<InternodeInfo>(child)) continue;
        commands.push_back({LSystemCommandType::Push, 0.0f});
        index++;
        ExportLSystemCommandsHelper(index, child, commands);
        commands.push_back({LSystemCommandType::Pop, 0.0f});
        index++;
    }
    if(!children.empty()){
        if (!scene->IsEntityValid(children[0]) || !scene->HasDataComponent<InternodeInfo>(children[0])) return;
        ExportLSystemCommandsHelper(index, children[0], commands);
    }
}

void Internode::OnInspect() {
    FileUtils::SaveFile("Export L-String", "L-System string", {".lstring"}, [&](const std::filesystem::path &path) {
        auto lString = ProjectManager::CreateTemporaryAsset<LSystemString>();
        ExportLString(lString);
        lString->Export(path);
    }, false);
    FileUtils::SaveFile("Export TreeIO file", "TreeIO format", {".tree"}, [&](const std::filesystem::path &path) {
        ExportTreeIOTree(path);
    }, false);
    if (ImGui::Button("Calculate L-String Indices")) {
        int index = 0;
        std::vector<LSystemCommand> commands;
        ExportLSystemCommandsHelper(index, GetOwner(), commands);
    }

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
}

void Internode::CollectAssetRef(std::vector<AssetRef> &list) {
}

void Internode::PostCloneAction(const std::shared_ptr<IPrivateComponent> &target) {

}

void Internode::Serialize(YAML::Emitter &out) {
    m_currentRoot.Save("m_currentRoot", out);
    m_apicalBud.Save("m_apicalBud", out);
    SaveList("m_lateralBuds", m_lateralBuds, out);
    out << YAML::Key << "m_fromApicalBud" << YAML::Value << m_fromApicalBud;
}

void Internode::Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) {
    m_currentRoot.Relink(map);
}

void Internode::Deserialize(const YAML::Node &in) {
    m_currentRoot.Load("m_currentRoot", in);
    m_apicalBud.Load("m_apicalBud", in);
    LoadList("m_lateralBuds", m_lateralBuds, in);
    m_fromApicalBud = in["m_fromApicalBud"].as<bool>();
}

void Internode::OnDestroy() {
    m_normalDir = glm::vec3(0, 0, 1);
    m_step = 4;
    m_foliageMatrices.clear();
    m_rings.clear();
    m_apicalBud.m_status = BudStatus::Sleeping;
    m_lateralBuds.clear();
    m_fromApicalBud = true;
    m_currentRoot.Clear();
    m_resource.reset();
}

Bound Internode::CalculateChildrenBound() {
    std::vector<Entity> internodes;
    CollectInternodes(internodes);
    Bound retVal;
    retVal.m_max = glm::vec3(-99999.0f);
    retVal.m_min = glm::vec3(99999.0f);
    auto scene = GetScene();
    for(const auto& i : internodes){
        auto position = scene->GetDataComponent<GlobalTransform>(i).GetPosition();
        retVal.m_min.x = glm::min(retVal.m_min.x, position.x);
        retVal.m_min.y = glm::min(retVal.m_min.y, position.y);
        retVal.m_min.z = glm::min(retVal.m_min.z, position.z);
        retVal.m_max.x = glm::max(retVal.m_max.x, position.x);
        retVal.m_max.y = glm::max(retVal.m_max.y, position.y);
        retVal.m_max.z = glm::max(retVal.m_max.z, position.z);
    }
    return retVal;
}

void Internode::ExportTreeIOTree(const std::filesystem::path &path) {
    ArrayTree tree;
    TreeNodeData rootNodeData;
    auto scene = GetScene();
    auto gt = scene->GetDataComponent<GlobalTransform>(GetOwner());
    auto internodeInfo = scene->GetDataComponent<InternodeInfo>(GetOwner());
    rootNodeData.direction = gt.GetRotation() * glm::vec3(0, 0, -1);
    rootNodeData.thickness = internodeInfo.m_thickness;
    rootNodeData.pos = gt.GetPosition();

    auto rootId = tree.addRoot(rootNodeData);
    auto children = scene->GetChildren(GetOwner());
    for(const auto& child : children) {
        ExportTreeIOTreeHelper(tree, child, rootId);
    }
    tree.saveTree(path.string());
}

void Internode::ExportTreeIOTreeHelper(ArrayTree& tree, const Entity &target, ArrayTreeT<TreeNodeData, TreeMetaData>::NodeIdT id) {
    TreeNodeData nodeData;
    auto scene = GetScene();
    auto gt = scene->GetDataComponent<GlobalTransform>(target);
    auto internodeInfo = scene->GetDataComponent<InternodeInfo>(target);
    nodeData.direction = gt.GetRotation() * glm::vec3(0, 0, -1);
    nodeData.thickness = internodeInfo.m_thickness;
    nodeData.pos = gt.GetPosition();

    auto currentId = tree.addNodeChild(id, nodeData);
    auto children = scene->GetChildren(target);
    for(const auto& child : children){
        ExportTreeIOTreeHelper(tree, child, currentId);
    }
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