//
// Created by lllll on 10/21/2022.
//

#include "PlantGrowth.hpp"

void Orchards::TreeGrowthModel::Grow() {
    if (!m_targetPlant) {
        UNIENGINE_ERROR("Plant not exist!");
        return;
    }
    auto sortedInternodeList = m_targetPlant->GetSortedInternodeList();
    for (const auto &internodeHandle: sortedInternodeList) {
        auto &internode = m_targetPlant->RefInternode(internodeHandle);
        if (internode.m_recycled) continue;
        //Pruning here.
    }
    for (auto it = sortedInternodeList.rbegin(); it != sortedInternodeList.rend(); it++) {
        auto &internode = m_targetPlant->RefInternode(*it);
        if (internode.m_recycled) continue;
        auto& internodeData = internode.m_data;
        internodeData.m_childTotalBiomass = 0;
        internodeData.m_maxDistanceToAnyBranchEnd = 0;
        if (internode.m_endNode) {
            //If current node is end node
            internodeData.m_inhibitor = 0;
        } else {
            //If current node is not end node
            int maxDistanceToAnyBranchEnd = 0;
            for (const auto &i: internode.m_children) {
                auto &childInternode = m_targetPlant->RefInternode(i);
                if (childInternode.m_endNode) {
                    internodeData.m_inhibitor += m_parameters.m_apicalDominanceBaseAgeDist.x *
                                                    glm::pow(
                                                            m_parameters.m_apicalDominanceBaseAgeDist.y,
                                                            childInternode.m_data.m_age);
                } else {
                    internodeData.m_inhibitor +=
                            childInternode.m_data.m_inhibitor * m_parameters.m_apicalDominanceBaseAgeDist.z;
                };
                internodeData.m_childTotalBiomass += childInternode.m_thickness;

                int childMaxDistanceToAnyBranchEnd =
                        childInternode.m_data.m_maxDistanceToAnyBranchEnd + 1;
                maxDistanceToAnyBranchEnd = glm::max(maxDistanceToAnyBranchEnd, childMaxDistanceToAnyBranchEnd);
            }
            internodeData.m_maxDistanceToAnyBranchEnd = maxDistanceToAnyBranchEnd;
            internodeData.m_sagging =
                    glm::min(
                            m_parameters.m_saggingFactorThicknessReductionMax.z,
                            m_parameters.m_saggingFactorThicknessReductionMax.x *
                                    internodeData.m_childTotalBiomass /
                            glm::pow(
                                    internode.m_thickness /
                                    m_parameters.m_endNodeThicknessAndControl.x,
                                    m_parameters.m_saggingFactorThicknessReductionMax.y));
        }
    }

    for (const auto &internodeHandle: sortedInternodeList) {
        auto &internode = m_targetPlant->RefInternode(internodeHandle);
        if (internode.m_recycled) continue;
        auto& internodeData = internode.m_data;
        internodeData.m_rootDistance = 0;
        if(internode.m_parent != -1){
            auto &parentInternode = m_targetPlant->RefInternode(internode.m_parent);
            internodeData.m_rootDistance = parentInternode.m_data.m_rootDistance + 1;
        }
    }
}

Orchards::TreeGrowthParameters::TreeGrowthParameters() {
    m_lateralBudCount = 2;
    m_branchingAngleMeanVariance = glm::vec2(30, 3);
    m_rollAngleMeanVariance = glm::vec2(120, 2);
    m_apicalAngleMeanVariance = glm::vec2(20, 2);
    m_gravitropism = -0.1f;
    m_phototropism = 0.05f;
    m_internodeLengthMeanVariance = glm::vec2(1, 0.1);
    m_endNodeThicknessAndControl = glm::vec2(0.01, 0.5);
    m_lateralBudFlushingProbability = 0.3f;
    m_neighborAvoidance = glm::vec3(0.05f, 1, 100);
    m_apicalControl = 2.0f;
    m_apicalDominanceBaseAgeDist = glm::vec3(0.12, 1, 0.3);
    m_lateralBudFlushingLightingFactor = 0.0f;
    m_budKillProbabilityApicalLateral = glm::vec2(0.0, 0.03);
    m_randomPruningOrderProtection = 1;
    m_randomPruningBaseAgeMax = glm::vec3(-0.1, 0.007, 0.5);
    m_lowBranchPruning = 0.15f;
    m_saggingFactorThicknessReductionMax = glm::vec3(6, 3, 0.5);
    m_matureAge = 30;
}
