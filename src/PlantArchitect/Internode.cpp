//
// Created by lllll on 8/27/2021.
//

#include "Internode.hpp"
#include "InternodeSystem.hpp"
#include "LSystemBehaviour.hpp"
#include "AssetManager.hpp"
using namespace PlantArchitect;

void Internode::Clone(const std::shared_ptr<IPrivateComponent> &target) {

}

void Internode::OnCreate() {
    m_internodeSystem = EntityManager::GetSystem<InternodeSystem>();
    m_branchMesh = AssetManager::CreateAsset<Mesh>();
    m_skinnedBranchMesh = AssetManager::CreateAsset<SkinnedMesh>();
    m_meshGenerated = false;
}

void Internode::OnRetrieve() {

}

void Internode::OnRecycle() {
    m_resource->Reset();
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
        target.ForEachChild([&](Entity child) {
            CollectInternodesHelper(child, results);
        });
    }
}

void Internode::CollectInternodes(std::vector<Entity> &results) {
    CollectInternodesHelper(GetOwner(), results);
}

void Internode::ExportLString(const std::shared_ptr<LString>& lString) {
    ExportLSystemCommandsHelper(GetOwner(), lString->commands);
}

void Internode::ExportLSystemCommandsHelper(const Entity &target, std::vector<LSystemCommand> &commands) {
    if (!target.IsValid() || !target.HasDataComponent<InternodeInfo>()) return;
    auto internodeInfo = target.GetDataComponent<InternodeInfo>();
    auto transform = target.GetDataComponent<Transform>();
    auto eulerRotation = transform.GetEulerRotation();
    if (eulerRotation.x != 0) {
        commands.push_back({LSystemCommandType::PitchUp, eulerRotation.x});
    }
    if (eulerRotation.y != 0) {
        commands.push_back({LSystemCommandType::TurnLeft, eulerRotation.y});
    }
    if (eulerRotation.z != 0) {
        commands.push_back({LSystemCommandType::RollLeft, eulerRotation.z});
    }
    commands.push_back({LSystemCommandType::Forward, internodeInfo.m_length});

    target.ForEachChild([&](Entity child){
        if (!child.IsValid() || !child.HasDataComponent<InternodeInfo>()) return;
        commands.push_back({LSystemCommandType::Push, 0.0f});
        ExportLSystemCommandsHelper(child, commands);
        commands.push_back({LSystemCommandType::Pop, 0.0f});
    });
}

void Internode::OnGui() {
    if(ImGui::Button("Generate L-String")){
        auto lString = AssetManager::CreateAsset<LString>();
        AssetManager::Share(lString);
        ExportLString(lString);
    }
}
