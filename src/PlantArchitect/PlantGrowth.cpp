//
// Created by lllll on 10/21/2022.
//

#include "PlantGrowth.hpp"

using namespace Orchards;

void ApplyTropism(const glm::vec3 &targetDir, float tropism, glm::vec3 &front, glm::vec3 &up) {
    const glm::vec3 dir = glm::normalize(targetDir);
    const float dotP = glm::abs(glm::dot(front, dir));
    if (dotP < 0.99f && dotP > -0.99f) {
        const glm::vec3 left = glm::cross(front, dir);
        const float maxAngle = glm::acos(dotP);
        const float rotateAngle = maxAngle * tropism;
        front = glm::normalize(
                glm::rotate(front, glm::min(maxAngle, rotateAngle), left));
        up = glm::normalize(glm::cross(glm::cross(front, up), front));
    }
}

void ApplyTropism(const glm::vec3 &targetDir, float tropism, glm::quat &rotation) {
    auto front = rotation * glm::vec3(0, 0, -1);
    auto up = rotation * glm::vec3(0, 1, 0);
    ApplyTropism(targetDir, tropism, front, up);
    rotation = glm::quatLookAt(front, up);
}

void TreeGrowthModel::CollectInhibitor(InternodeHandle internodeHandle) {
    auto &internode = m_targetPlant->RefInternode(internodeHandle);
    auto &internodeData = internode.m_data;
    if (internode.m_endNode) {
        //If current node is end node
        internodeData.m_inhibitor = 0;
    } else {
        //If current node is not end node
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
        }
    }
}

void TreeGrowthModel::GrowInternode(InternodeHandle internodeHandle) {
    //Internode elongation.
    if (m_targetPlant->RefInternode(internodeHandle).m_endNode) {
        auto newInternodeHandle = m_targetPlant->Extend(internodeHandle, false);
        auto &oldInternode = m_targetPlant->RefInternode(internodeHandle);
        auto &newInternode = m_targetPlant->RefInternode(newInternodeHandle);

        newInternode.m_length = 1.0f;
        newInternode.m_localRotation = glm::linearRand(glm::vec3(0.0f), glm::vec3(360.0f));

        newInternode.m_thickness = oldInternode.m_thickness * 0.9f;
    } else if (m_targetPlant->RefInternode(internodeHandle).m_handle % 3 == 0) {
        auto newInternodeHandle = m_targetPlant->Extend(internodeHandle, true);
        auto &newInternode = m_targetPlant->RefInternode(newInternodeHandle);
        auto &oldInternode = m_targetPlant->RefInternode(internodeHandle);

        newInternode.m_length = 1.0f;
        newInternode.m_localRotation = glm::linearRand(glm::vec3(0.0f), glm::vec3(360.0f));

        newInternode.m_thickness = oldInternode.m_thickness * 0.9f;
    }
}

void TreeGrowthModel::CalculateSagging(InternodeHandle internodeHandle) {
    auto &internode = m_targetPlant->RefInternode(internodeHandle);
    auto &internodeData = internode.m_data;
    if (!internode.m_endNode) {
        //If current node is not end node
        float maxDistanceToAnyBranchEnd = 0;
        for (const auto &i: internode.m_children) {
            auto &childInternode = m_targetPlant->RefInternode(i);
            internodeData.m_childTotalBiomass += childInternode.m_thickness * childInternode.m_length;
            float childMaxDistanceToAnyBranchEnd =
                    childInternode.m_data.m_maxDistanceToAnyBranchEnd + childInternode.m_length;
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

void TreeGrowthModel::Grow() {
    if (!m_initialized) {
        UNIENGINE_ERROR("Not initialized!");
        return;
    }

    if (!m_targetPlant) {
        UNIENGINE_ERROR("Plant not exist!");
        return;
    }
#pragma region Preprocess
    m_targetPlant->SortLists();
    {
        const auto &sortedInternodeList = m_targetPlant->GetSortedInternodeList();
        const auto &sortedBranchList = m_targetPlant->GetSortedBranchList();
        for (const auto &internodeHandle: sortedInternodeList) {
            auto &internode = m_targetPlant->RefInternode(internodeHandle);
            //Pruning here.
            if (internode.m_recycled) continue;
            if (internode.m_handle > 10 && glm::linearRand(0, 10) == 0) m_targetPlant->PruneInternode(internodeHandle);
        }
    }
#pragma endregion
#pragma region Grow
    m_targetPlant->SortLists();
    {
        const auto &sortedInternodeList = m_targetPlant->GetSortedInternodeList();
        const auto &sortedBranchList = m_targetPlant->GetSortedBranchList();
        for (auto it = sortedInternodeList.rbegin(); it != sortedInternodeList.rend(); it++) {
            auto internodeHandle = *it;
            CollectInhibitor(internodeHandle);
            GrowInternode(internodeHandle);
            CalculateSagging(internodeHandle);
        }

    }
#pragma endregion
#pragma region Postprocess
    m_targetPlant->SortLists();
    {
        const auto &sortedInternodeList = m_targetPlant->GetSortedInternodeList();
        const auto &sortedBranchList = m_targetPlant->GetSortedBranchList();
        for (const auto &internodeHandle: sortedInternodeList) {
            auto &internode = m_targetPlant->RefInternode(internodeHandle);
            auto &internodeData = internode.m_data;
            if (internode.m_parent == -1) {
                internode.m_globalPosition = glm::vec3(0.0f);
                internode.m_globalRotation = glm::vec3(0.0f);

                internodeData.m_rootDistance = internode.m_length;
                internodeData.m_apicalControl = internodeData.m_childTotalBiomass;
            } else {
                auto &parentInternode = m_targetPlant->RefInternode(internode.m_parent);
                internodeData.m_rootDistance = parentInternode.m_data.m_rootDistance + internode.m_length;

                internode.m_globalRotation = parentInternode.m_globalRotation * internode.m_localRotation;
                internode.m_globalPosition = parentInternode.m_globalPosition + parentInternode.m_length *
                                                                                (parentInternode.m_globalRotation *
                                                                                 glm::vec3(0, 0, -1));
#pragma region Apply Sagging
                auto parentGlobalRotation = m_targetPlant->RefInternode(internode.m_parent).m_globalRotation;
                internode.m_globalRotation = parentGlobalRotation * internode.m_localRotation;
                auto front = internode.m_globalRotation * glm::vec3(0, 0, -1);
                auto up = internode.m_globalRotation * glm::vec3(0, 1, 0);
                float dotP = glm::abs(glm::dot(front, m_gravityDirection));
                ApplyTropism(-m_gravityDirection, internodeData.m_sagging * (1.0f - dotP), front, up);
                internode.m_globalRotation = glm::quatLookAt(front, up);
                internode.m_localRotation = glm::inverse(parentGlobalRotation) * internode.m_globalRotation;
#pragma endregion
            }

            float apicalControl = glm::pow(m_parameters.m_apicalControlBaseDistFactor.x, glm::max(1.0f, 1.0f /
                                                                                                        internodeData.m_rootDistance *
                                                                                                        m_parameters.m_apicalControlBaseDistFactor.y));
            float totalApicalControl = 0.0f;
            for (const auto &i: internode.m_children) {
                auto &childInternode = m_targetPlant->RefInternode(i);
                auto &childInternodeData = childInternode.m_data;
                childInternodeData.m_apicalControl = glm::pow(
                        childInternodeData.m_childTotalBiomass + childInternode.m_length * childInternode.m_thickness,
                        apicalControl);
                totalApicalControl += childInternodeData.m_apicalControl;
            }
            for (const auto &i: internode.m_children) {
                auto &childInternode = m_targetPlant->RefInternode(i);
                auto &childInternodeData = childInternode.m_data;
                childInternodeData.m_apicalControl =
                        internodeData.m_apicalControl * childInternodeData.m_apicalControl / totalApicalControl;
            }
        }
    }
#pragma endregion
}

void TreeGrowthModel::Initialize() {
    m_targetPlant = std::make_shared<Plant<BranchData, InternodeData>>();
    m_initialized = true;
}

void TreeGrowthModel::Clear() {
    m_targetPlant.reset();
    m_initialized = false;
}


TreeGrowthParameters::TreeGrowthParameters() {
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
    m_apicalControlBaseDistFactor = {2.0f, 0.95f};
    m_apicalDominanceBaseAgeDist = glm::vec3(0.12, 1, 0.3);
    m_lateralBudFlushingLightingFactor = 0.0f;
    m_budKillProbabilityApicalLateral = glm::vec2(0.0, 0.03);
    m_randomPruningOrderProtection = 1;
    m_randomPruningBaseAgeMax = glm::vec3(-0.1, 0.007, 0.5);
    m_lowBranchPruning = 0.15f;
    m_saggingFactorThicknessReductionMax = glm::vec3(6, 3, 0.5);
    m_matureAge = 30;
}
