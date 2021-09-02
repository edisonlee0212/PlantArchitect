//
// Created by lllll on 9/1/2021.
//

#include "SpaceColonizationTreeToLString.hpp"
#include "EntityManager.hpp"
#include "InternodeSystem.hpp"
#include "AssetManager.hpp"
#include "LSystemBehaviour.hpp"
#include "IVolume.hpp"
using namespace Scripts;

void SpaceColonizationTreeToLString::OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) {
    if(m_remainingInstanceAmount <= 0){
        m_remainingInstanceAmount = 0;
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }
    m_remainingGrowthIterations = m_perTreeGrowthIteration;
    pipeline.m_status = AutoTreeGenerationPipelineStatus::Growth;
    if(m_spaceColonizationTreeBehaviour.expired()){
        m_spaceColonizationTreeBehaviour = EntityManager::GetSystem<InternodeSystem>()->GetInternodeBehaviour<SpaceColonizationBehaviour>();
    }
    auto behaviour = m_spaceColonizationTreeBehaviour.lock();
    for (int i = 0; i < m_attractionPointAmount; i++) {
        behaviour->m_attractionPoints.push_back(behaviour->m_volumes[0].Get<IVolume>()->GetRandomPoint());
    }
    m_currentGrowingTree = behaviour->NewPlant(SpaceColonizationParameters(), Transform());
}

void SpaceColonizationTreeToLString::OnGrowth(AutoTreeGenerationPipeline& pipeline) {
    if(m_remainingInstanceAmount == 0){
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }
    auto internodeSystem = EntityManager::GetSystem<InternodeSystem>();
    internodeSystem->Simulate(1.0f);
    m_remainingGrowthIterations--;
    if(m_remainingGrowthIterations == 0){
        pipeline.m_status = AutoTreeGenerationPipelineStatus::AfterGrowth;
    }
}

void SpaceColonizationTreeToLString::OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) {
    auto lString = AssetManager::CreateAsset<LString>();
    m_currentGrowingTree.GetOrSetPrivateComponent<Internode>().lock()->ExportLString(lString);
    //path here
    lString->Save(m_currentExportFolder / (std::to_string(m_generationAmount - m_remainingInstanceAmount) + ".lstring"));
    auto behaviour = m_spaceColonizationTreeBehaviour.lock();
    behaviour->Recycle(m_currentGrowingTree);
    behaviour->m_attractionPoints.clear();
    m_remainingInstanceAmount--;
    if(m_remainingInstanceAmount == 0){
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
    }else{
        pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
    }
}

void SpaceColonizationTreeToLString::OnInspect() {
    ImGui::DragInt("Generation Amount", &m_generationAmount);
    ImGui::DragInt("Growth iteration", &m_perTreeGrowthIteration);
    ImGui::DragInt("Attraction point per plant", &m_attractionPointAmount);
    if(m_remainingInstanceAmount == 0) {
        if (ImGui::Button("Start")) {
            std::filesystem::create_directories(m_currentExportFolder);
            m_remainingInstanceAmount = m_generationAmount;
        }
    }else{
        ImGui::Text("Task dispatched...");
    }
}

void SpaceColonizationTreeToLString::OnIdle(AutoTreeGenerationPipeline& pipeline) {
    if(m_remainingInstanceAmount > 0){
        pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
        return;
    }
}

