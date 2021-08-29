//
// Created by lllll on 8/27/2021.
//

#include "InternodeSystem.hpp"
#include "DefaultInternodeBehaviour.hpp"
using namespace PlantArchitect;

void InternodeSystem::OnInspect() {
    if(ImGui::Button("Test")){
        Simulate(1.0f);
    }

    if(ImGui::TreeNode("Internode Behaviours")){
        BehaviourSlotButton();
        int index = 0;
        bool skip = false;
        for(auto& i : m_internodeBehaviours){
            if(EditorManager::DragAndDropButton<IAsset>(i, "Slot " + std::to_string(index++))){
                skip = true;
                break;
            }
        }
        if(skip){
            int index = 0;
            for(auto& i : m_internodeBehaviours){
                if(!i.Get<IInternodeBehaviour>()){
                    m_internodeBehaviours.erase(m_internodeBehaviours.begin() + index);
                    break;
                }
                index++;
            }
        }
        ImGui::TreePop();
    }
}

void InternodeSystem::OnCreate() {
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeTag());
    Enable();
}

void InternodeSystem::Simulate(float deltaTime) {
    //0. Pre processing
    for (auto &i: m_internodeBehaviours) {
        auto behaviour = i.Get<IInternodeBehaviour>();
        if (behaviour) behaviour->PreProcess();
    }

    //1. Collect resource from environment
    EntityManager::ForEach<InternodeTag>(JobManager::PrimaryWorkers(), m_internodesQuery,
                                         [=](int i, Entity entity, InternodeTag &tag) {
                                             entity.GetOrSetPrivateComponent<Internode>().lock()->CollectResource(
                                                     deltaTime);
                                         }, true);
    //2. Upstream resource
    EntityManager::ForEach<InternodeTag>(JobManager::PrimaryWorkers(), m_internodesQuery,
                                         [=](int i, Entity entity, InternodeTag &tag) {
                                             entity.GetOrSetPrivateComponent<Internode>().lock()->UpStreamResource(
                                                     deltaTime);
                                         }, true);
    //3. Downstream resource
    EntityManager::ForEach<InternodeTag>(JobManager::PrimaryWorkers(), m_internodesQuery,
                                         [=](int i, Entity entity, InternodeTag &tag) {
                                             entity.GetOrSetPrivateComponent<Internode>().lock()->DownStreamResource(
                                                     deltaTime);
                                         }, true);

    //4. Growth
    for (auto &i: m_internodeBehaviours) {
        auto behaviour = i.Get<IInternodeBehaviour>();
        if (behaviour) behaviour->Grow();
    }

    //5. Post processing
    for (auto &i: m_internodeBehaviours) {
        auto behaviour = i.Get<IInternodeBehaviour>();
        if (behaviour) behaviour->PostProcess();
    }
}

void InternodeSystem::BehaviourSlotButton() {
    ImGui::Text("Drop Behaviour");
    ImGui::SameLine();
    ImGui::Button("Here");
    if (ImGui::BeginDragDropTarget())
    {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DefaultInternodeBehaviour"))
        {
            IM_ASSERT(payload->DataSize == sizeof(std::shared_ptr<IAsset>));
            std::shared_ptr<DefaultInternodeBehaviour> payload_n =
                    std::dynamic_pointer_cast<DefaultInternodeBehaviour>(*static_cast<std::shared_ptr<IAsset> *>(payload->Data));
            if (payload_n.get())
            {
                bool search = false;
                for(auto& i : m_internodeBehaviours){
                    if(i.Get<IInternodeBehaviour>()->GetTypeName() == "DefaultInternodeBehaviour") search = true;
                }
                if(!search){
                    m_internodeBehaviours.emplace_back(payload_n);
                }
            }
        }
        ImGui::EndDragDropTarget();
    }
}


