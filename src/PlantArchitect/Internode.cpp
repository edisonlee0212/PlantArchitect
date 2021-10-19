//
// Created by lllll on 8/27/2021.
//

#include "Internode.hpp"
#include "InternodeSystem.hpp"
#include "LSystemBehaviour.hpp"
#include "AssetManager.hpp"
#include "InternodeFoliage.hpp"
using namespace PlantArchitect;

void Internode::OnCreate() {
    m_internodeSystem = EntityManager::GetCurrentScene()->GetSystem<InternodeSystem>();
 }

void Internode::OnRetrieve() {

}

void Internode::OnRecycle() {
    m_resource->Reset();
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
    auto owner = GetOwner();
    m_resource->DownStream(deltaTime, owner, owner.GetParent());
}

void Internode::UpStreamResource(float deltaTime) {
    auto owner = GetOwner();
    auto children = owner.GetChildren();
    for (const auto &child: children) {
        m_resource->UpStream(deltaTime, owner, child);
    }
}

void Internode::CollectResource(float deltaTime) {
    m_resource->Collect(deltaTime, GetOwner());
}

void Internode::CollectInternodesHelper(const Entity &target, std::vector<Entity> &results) {
    if (target.IsValid() && target.HasDataComponent<InternodeInfo>() && target.HasPrivateComponent<Internode>()) {
        results.push_back(target);
        target.ForEachChild([&](const std::shared_ptr<Scene>& scene, Entity child) {
            CollectInternodesHelper(child, results);
        });
    }
}

void Internode::CollectInternodes(std::vector<Entity> &results) {
    CollectInternodesHelper(GetOwner(), results);
}

void Internode::ExportLString(const std::shared_ptr<LString>& lString) {
    int index = 0;
    ExportLSystemCommandsHelper(index, GetOwner(), lString->commands);
}

void Internode::ExportLSystemCommandsHelper(int& index, const Entity &target, std::vector<LSystemCommand> &commands) {
    if (!target.IsValid() || !target.HasDataComponent<InternodeInfo>()) return;
    auto internodeInfo = target.GetDataComponent<InternodeInfo>();
    auto transform = target.GetDataComponent<Transform>();
    auto eulerRotation = transform.GetEulerRotation();
    if (eulerRotation.x > 0) {
        commands.push_back({LSystemCommandType::PitchUp, eulerRotation.x});
        index++;
    }else if(eulerRotation.x < 0){
        commands.push_back({LSystemCommandType::PitchDown, -eulerRotation.x});
        index++;
    }
    if (eulerRotation.y > 0) {
        commands.push_back({LSystemCommandType::TurnLeft, eulerRotation.y});
        index++;
    }else if(eulerRotation.y < 0){
        commands.push_back({LSystemCommandType::TurnRight, -eulerRotation.y});
        index++;
    }
    if (eulerRotation.z > 0) {
        commands.push_back({LSystemCommandType::RollLeft, eulerRotation.z});
        index++;
    }else if(eulerRotation.z < 0){
        commands.push_back({LSystemCommandType::RollRight, -eulerRotation.z});
        index++;
    }
    commands.push_back({LSystemCommandType::Forward, internodeInfo.m_length});
    internodeInfo.m_index = index;
    target.SetDataComponent(internodeInfo);
    index++;

    target.ForEachChild([&](const std::shared_ptr<Scene>& scene, Entity child){
        if (!child.IsValid() || !child.HasDataComponent<InternodeInfo>()) return;
        commands.push_back({LSystemCommandType::Push, 0.0f});
        index++;
        ExportLSystemCommandsHelper(index, child, commands);
        commands.push_back({LSystemCommandType::Pop, 0.0f});
        index++;
    });
}

void Internode::OnInspect() {
    if(ImGui::Button("Generate L-String")){
        auto lString = AssetManager::CreateAsset<LString>();
        AssetManager::Share(lString);
        ExportLString(lString);
    }
    if(ImGui::Button("Calculate L-String Indices")){
        int index = 0;
        std::vector<LSystemCommand> commands;
        ExportLSystemCommandsHelper(index, GetOwner(), commands);
    }
    m_foliage.Get<InternodeFoliage>()->OnInspect();

    if(ImGui::TreeNodeEx("Buds")){
        ImGui::Text("Apical bud:");
        m_apicalBud.OnInspect();
        ImGui::Text("Lateral bud:");
        for(int i = 0; i < m_lateralBuds.size(); i++){
            if(ImGui::TreeNodeEx(("Bud " + std::to_string(i)).c_str())){
                m_lateralBuds[i].OnInspect();
                ImGui::TreePop();
            }
        }

        ImGui::TreePop();
    }
}

void Bud::OnInspect() {
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
