//
// Created by lllll on 9/1/2021.
//

#include "SpaceColonizationTreeToLString.hpp"
#include "EntityManager.hpp"
#include "InternodeSystem.hpp"
#include "EmptyInternodeResource.hpp"
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
    m_currentGrowingTree = behaviour->Retrieve();
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
    if(m_remainingInstanceAmount == 0){
        pipeline.m_status = AutoTreeGenerationPipelineStatus::Idle;
        return;
    }


    m_remainingInstanceAmount--;
    pipeline.m_status = AutoTreeGenerationPipelineStatus::BeforeGrowth;
}

void SpaceColonizationTreeToLString::OnInspect() {

}
