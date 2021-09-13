//
// Created by lllll on 8/29/2021.
//

#include "GeneralTreeBehaviour.hpp"
#include "InternodeSystem.hpp"
#include <DefaultInternodeResource.hpp>
#include "EmptyInternodeResource.hpp"
#include "TransformManager.hpp"

using namespace PlantArchitect;

void GeneralTreeBehaviour::Grow(int iteration) {
    if (iteration == 0) {
        m_currentPlants.clear();
        CollectRoots(m_internodesQuery, m_currentPlants);
    }
    int plantSize = m_currentPlants.size();
#pragma region PreProcess
#pragma region InternodeStatus
    ParallelForEachRoot(m_currentPlants, [&](int plantIndex, Entity root) {
        if(!root.GetDataComponent<InternodeInfo>().m_isRealRoot) return;
        auto internodeInfo = root.GetDataComponent<InternodeInfo>();
        auto internodeStatus = root.GetDataComponent<InternodeStatus>();
        internodeInfo.m_currentRoot = root;
        internodeStatus.m_distanceToRoot = 0;
        internodeInfo.m_endNode = false;
        internodeStatus.m_biomass = internodeInfo.m_length * internodeInfo.m_thickness * internodeInfo.m_thickness;
        root.SetDataComponent(internodeInfo);
        root.SetDataComponent(internodeStatus);
        TreeGraphWalker(root, root,
                        [&](Entity parent, Entity child) {
                            auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                            auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
                            auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                            auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                            //Low branch pruning.
                            if (childInternodeStatus.m_order != 0) {
                                auto childParameter = child.GetDataComponent<GeneralTreeParameters>();
                                if (childInternodeStatus.m_distanceToRoot /
                                    internodeStatus.m_maxDistanceToAnyBranchEnd <
                                    childParameter.m_lowBranchPruning) {
                                    Recycle(child);
                                    return;
                                }
                            }
                            auto childInternode = child.GetOrSetPrivateComponent<Internode>().lock();
                            childInternodeInfo.m_currentRoot = root;
                            childInternodeStatus.m_distanceToRoot =
                                    parentInternodeInfo.m_length + parentInternodeStatus.m_distanceToRoot;
                            childInternodeStatus.m_biomass =
                                    childInternodeInfo.m_length * childInternodeInfo.m_thickness;
                            if (!childInternode->m_fromApicalBud) {
                                childInternodeStatus.m_order = parentInternodeStatus.m_order + 1;
                            } else {
                                childInternodeStatus.m_order = parentInternodeStatus.m_order;
                            }
                            child.SetDataComponent(childInternodeInfo);
                            child.SetDataComponent(childInternodeStatus);
                        },
                        [&](Entity parent) {
                            auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                            auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
                            auto parentParameters = parent.GetDataComponent<GeneralTreeParameters>();
                            parentInternodeInfo.m_endNode = false;
                            parentInternodeStatus.m_inhibitor = 0;
                            parentInternodeStatus.m_totalDistanceToAllBranchEnds = parentInternodeStatus.m_childTotalBiomass = 0;
                            float maxDistanceToAnyBranchEnd = -1.0f;
                            float maxTotalDistanceToAllBranchEnds = -1.0f;
                            float maxChildTotalBiomass = -1.0f;
                            parent.ForEachChild([&](Entity child) {
                                auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                                if (childInternodeInfo.m_endNode) {
                                    auto endNodeParameters = child.GetDataComponent<GeneralTreeParameters>();
                                    auto endNodeInternode = child.GetOrSetPrivateComponent<Internode>().lock();
                                    float randomFactor = glm::min(endNodeParameters.m_randomPruningBaseAgeMax.z,
                                                                  endNodeParameters.m_randomPruningBaseAgeMax.x +
                                                                  endNodeParameters.m_randomPruningBaseAgeMax.y *
                                                                  endNodeInternode->m_age);
                                    if (childInternodeStatus.m_order >
                                        endNodeParameters.m_randomPruningOrderProtection &&
                                        randomFactor > glm::linearRand(0.0f, 1.0f)) {
                                        RecycleSingle(child);
                                        return;
                                    }
                                    parentInternodeStatus.m_inhibitor +=
                                            endNodeParameters.m_apicalDominanceBaseAgeDist.x *
                                            glm::pow(endNodeParameters.m_apicalDominanceBaseAgeDist.y,
                                                     endNodeInternode->m_age);
                                } else {
                                    parentInternodeStatus.m_inhibitor +=
                                            childInternodeStatus.m_inhibitor *
                                            glm::pow(parentParameters.m_apicalDominanceBaseAgeDist.z,
                                                     parentInternodeInfo.m_length);
                                }

                                float childTotalDistanceToAllBranchEnds =
                                        childInternodeStatus.m_totalDistanceToAllBranchEnds +
                                        childInternodeInfo.m_length;
                                float childTotalBiomass =
                                        childInternodeStatus.m_childTotalBiomass + childInternodeStatus.m_biomass;
                                float childMaxDistanceToAnyBranchEnd =
                                        childInternodeStatus.m_maxDistanceToAnyBranchEnd +
                                        childInternodeInfo.m_length;
                                parentInternodeStatus.m_totalDistanceToAllBranchEnds += childTotalDistanceToAllBranchEnds;
                                parentInternodeStatus.m_childTotalBiomass += childTotalBiomass;
                                if (maxTotalDistanceToAllBranchEnds < childTotalDistanceToAllBranchEnds) {
                                    maxTotalDistanceToAllBranchEnds = childTotalDistanceToAllBranchEnds;
                                    parentInternodeStatus.m_largestChild = child;
                                }
                                if (maxDistanceToAnyBranchEnd < childMaxDistanceToAnyBranchEnd) {
                                    maxDistanceToAnyBranchEnd = childMaxDistanceToAnyBranchEnd;
                                    parentInternodeStatus.m_longestChild = child;
                                }
                                if (maxChildTotalBiomass < childTotalBiomass) {
                                    maxChildTotalBiomass = childTotalBiomass;
                                    parentInternodeStatus.m_heaviestChild = child;
                                }
                            });
                            parentInternodeStatus.m_maxDistanceToAnyBranchEnd = maxDistanceToAnyBranchEnd;
                            parentInternodeStatus.m_sagging =
                                    glm::min(parentParameters.m_saggingFactorThicknessReductionMax.z,
                                             parentParameters.m_saggingFactorThicknessReductionMax.x *
                                             parentInternodeStatus.m_childTotalBiomass /
                                             glm::pow(parentInternodeInfo.m_thickness /
                                                      parentParameters.m_endNodeThicknessAndControl.x,
                                                      parentParameters.m_saggingFactorThicknessReductionMax.y));
                            parent.SetDataComponent(parentInternodeStatus);
                            parent.SetDataComponent(parentInternodeInfo);
                        },
                        [&](Entity endNode) {
                            auto endNodeInternodeInfo = endNode.GetDataComponent<InternodeInfo>();
                            auto endNodeInternodeStatus = endNode.GetDataComponent<InternodeStatus>();
                            endNodeInternodeInfo.m_endNode = true;
                            endNodeInternodeStatus.m_inhibitor = 0.0f;
                            endNodeInternodeStatus.m_maxDistanceToAnyBranchEnd = endNodeInternodeStatus.m_totalDistanceToAllBranchEnds = endNodeInternodeStatus.m_childTotalBiomass = 0;
                            endNodeInternodeStatus.m_largestChild = endNodeInternodeStatus.m_longestChild = endNodeInternodeStatus.m_heaviestChild = Entity();
                            endNode.SetDataComponent(endNodeInternodeInfo);
                            endNode.SetDataComponent(endNodeInternodeStatus);
                        });
    });
    ParallelForEachRoot(m_currentPlants, [&](int plantIndex, Entity root) {
        TreeGraphWalkerRootToEnd(root, root, [](Entity parent, Entity child) {
            auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
            auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
            if (parentInternodeStatus.m_largestChild == child) {
                childInternodeStatus.m_level = parentInternodeStatus.m_level;
            } else {
                childInternodeStatus.m_level = parentInternodeStatus.m_level + 1;
            }
            child.SetDataComponent(childInternodeStatus);
        });
    });
#pragma endregion
#pragma region Illumination
    EntityManager::ForEach<InternodeInfo, InternodeIllumination>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeTag, InternodeIllumination &internodeIllumination) {
                 internodeIllumination.m_direction = glm::vec3(0, 1, 0);
                 internodeIllumination.m_intensity = 1.0f;
             }, true);
#pragma endregion
#pragma region Water
#pragma region Collect water requests
    EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination) {
                 if(!internodeInfo.m_currentRoot.GetDataComponent<InternodeInfo>().m_isRealRoot) return;
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 if (internode->m_apicalBud.m_status == BudStatus::Sleeping) {
                     internodeWaterPressure.m_value = 1;
                 } else {
                     internodeWaterPressure.m_value = 0;
                 }
             }, true);
    auto workerSize = JobManager::PrimaryWorkers().Size();
    std::vector<std::vector<float>> totalRequestCollector;
    totalRequestCollector.resize(workerSize);
    for (auto &i: totalRequestCollector) {
        i.resize(plantSize);
        for (auto &j: i) j = 0;
    }
    EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater, GeneralTreeParameters>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                 InternodeWater &internodeWater,
                 GeneralTreeParameters &generalTreeParameters) {
                 if(!internodeInfo.m_currentRoot.GetDataComponent<InternodeInfo>().m_isRealRoot) return;
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 int plantIndex = 0;
                 for (const auto &plant: m_currentPlants) {
                     if (internodeInfo.m_currentRoot == plant) {
                         float apicalControl = generalTreeParameters.m_apicalControlBaseAge.x *
                                               glm::pow(generalTreeParameters.m_apicalControlBaseAge.y,
                                                        internode->m_age);
                         totalRequestCollector[i % workerSize][plantIndex] +=
                                 glm::pow(1.0f / glm::max(1.0f, apicalControl), internodeStatus.m_level) *
                                 internodeWaterPressure.m_value *
                                 internodeIllumination.m_intensity;
                         break;
                     }
                     plantIndex++;
                 }
             }, true);
    std::vector<float> totalRequests;
    totalRequests.resize(plantSize);
    for (const auto &i: totalRequestCollector) {
        for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
            totalRequests[plantIndex] += i[plantIndex];
        }
    }
#pragma endregion

#pragma region Collect and distribute water
    std::vector<std::vector<float>> totalWaterCollector;
    totalWaterCollector.resize(workerSize);
    for (auto &i: totalWaterCollector) {
        i.resize(plantSize);
        for (auto &j: i) j = 0;
    }
    EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                 InternodeWater &internodeWater) {
                 if(!internodeInfo.m_currentRoot.GetDataComponent<InternodeInfo>().m_isRealRoot) return;
                 int plantIndex = 0;
                 for (const auto &plant: m_currentPlants) {
                     if (internodeInfo.m_currentRoot == plant) {
                         totalWaterCollector[i % workerSize][plantIndex] += internodeWater.m_value;
                         break;
                     }
                     plantIndex++;
                 }
             }, true);
    std::vector<float> totalWater;
    totalWater.resize(plantSize);
    for (const auto &i: totalWaterCollector) {
        for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
            totalWater[plantIndex] += i[plantIndex];
        }
    }
    std::vector<float> waterDividends;
    waterDividends.resize(plantSize);
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        waterDividends[plantIndex] = 1.0f;// totalWater[plantIndex] / totalRequests[plantIndex];
        if (totalRequests[plantIndex] == 0) waterDividends[plantIndex] = 0;
    }
    EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater, GeneralTreeParameters>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                 InternodeWater &internodeWater,
                 GeneralTreeParameters &generalTreeParameters) {
                 if(!internodeInfo.m_currentRoot.GetDataComponent<InternodeInfo>().m_isRealRoot) return;
                 int plantIndex = 0;
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 for (const auto &plant: m_currentPlants) {
                     if (internodeInfo.m_currentRoot == plant) {
                         float apicalControl = generalTreeParameters.m_apicalControlBaseAge.x *
                                               glm::pow(generalTreeParameters.m_apicalControlBaseAge.y,
                                                        internode->m_age);
                         internodeWater.m_value =
                                 glm::pow(1.0f / glm::max(1.0f, apicalControl), internodeStatus.m_level) *
                                 waterDividends[plantIndex] * internodeWaterPressure.m_value *
                                 internodeIllumination.m_intensity;
                         break;
                     }
                     plantIndex++;
                 }
             }, true);
#pragma endregion
#pragma endregion
#pragma endregion
#pragma region Main Growth
    EntityManager::ForEach<Transform, GlobalTransform, InternodeInfo, InternodeStatus, InternodeWater, InternodeIllumination, GeneralTreeParameters>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity,
                 Transform &transform, GlobalTransform &globalTransform,
                 InternodeInfo &internodeInfo, InternodeStatus &internodeStatus,
                 InternodeWater &internodeWater, InternodeIllumination &internodeIllumination,
                 GeneralTreeParameters &generalTreeParameters) {
                 if(!internodeInfo.m_currentRoot.GetDataComponent<InternodeInfo>().m_isRealRoot) return;
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 internode->m_age++;
                 //0. Apply sagging here.
                 auto parent = entity.GetParent();
                 if (!parent.IsNull()) {
                     auto parentGlobalRotation = parent.GetDataComponent<GlobalTransform>().GetRotation();
                     glm::quat desiredGlobalRotation = parentGlobalRotation * internodeStatus.m_desiredLocalRotation;
                     glm::vec3 desiredGlobalFront = desiredGlobalRotation * glm::vec3(0, 0, -1);
                     glm::vec3 desiredGlobalUp = desiredGlobalRotation * glm::vec3(0, 1, 0);
                     float dotP = glm::abs(glm::dot(desiredGlobalFront, glm::vec3(0, 1, 0)));
                     ApplyTropism(glm::vec3(0, -1, 0), (internodeStatus.m_sagging * (1.0f - dotP)), desiredGlobalFront,
                                  desiredGlobalUp);
                     desiredGlobalRotation = glm::quatLookAt(desiredGlobalFront, desiredGlobalUp);
                     internodeInfo.m_localRotation = glm::inverse(parentGlobalRotation) * desiredGlobalRotation;
                 }

                 //1. Internode elongation.
                 switch (internode->m_apicalBud.m_status) {
                     case BudStatus::Sleeping: {
                         if (generalTreeParameters.m_budKillProbabilityApicalLateral.x >
                             glm::linearRand(0.0f, 1.0f)) {
                             internode->m_apicalBud.m_status = BudStatus::Died;
                             break;
                         }
                         if (internodeWater.m_value == 0) return;
                         float desiredLength = glm::gaussRand(generalTreeParameters.m_internodeLengthMeanVariance.x,
                                                              generalTreeParameters.m_internodeLengthMeanVariance.y);
                         internodeInfo.m_length += internodeWater.m_value;
                         internodeWater.m_value = 0;
                         if (internodeInfo.m_length > desiredLength) {
                             internodeInfo.m_endNode = false;
                             internodeWater.m_value = internodeInfo.m_length - desiredLength;
                             internodeInfo.m_length = desiredLength;
                             internode->m_apicalBud.m_newInternodeInfo = InternodeInfo();
                             internode->m_apicalBud.m_newInternodeInfo.m_thickness = generalTreeParameters.m_endNodeThicknessAndControl.x;
                             internode->m_apicalBud.m_newInternodeInfo.m_currentRoot = internodeInfo.m_currentRoot;
                             glm::quat desiredGlobalRotation = globalTransform.GetRotation();
                             glm::vec3 desiredGlobalFront = desiredGlobalRotation * glm::vec3(0, 0, -1);
                             glm::vec3 desiredGlobalUp = desiredGlobalRotation * glm::vec3(0, 1, 0);
                             desiredGlobalUp = glm::rotate(desiredGlobalUp, glm::radians(
                                                                   glm::gaussRand(generalTreeParameters.m_rollAngleMeanVariance.x,
                                                                                  generalTreeParameters.m_rollAngleMeanVariance.y)),
                                                           desiredGlobalFront);
                             desiredGlobalFront = glm::rotate(desiredGlobalFront, glm::radians(
                                                                      glm::gaussRand(generalTreeParameters.m_apicalAngleMeanVariance.x,
                                                                                     generalTreeParameters.m_apicalAngleMeanVariance.y)),
                                                              desiredGlobalUp);
                             ApplyTropism(glm::vec3(0, -1, 0), generalTreeParameters.m_gravitropism,
                                          desiredGlobalFront, desiredGlobalUp);
                             ApplyTropism(internodeIllumination.m_direction, generalTreeParameters.m_phototropism,
                                          desiredGlobalFront, desiredGlobalUp);

                             desiredGlobalRotation = glm::quatLookAt(desiredGlobalFront, desiredGlobalUp);
                             internode->m_apicalBud.m_newInternodeInfo.m_localRotation =
                                     glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
                             internode->m_apicalBud.m_status = BudStatus::Flushing;
                             //Form lateral buds here.
                             float turnAngle = glm::radians(360.0f / generalTreeParameters.m_lateralBudCount);
                             for (int lateralBudIndex = 0;
                                  lateralBudIndex < generalTreeParameters.m_lateralBudCount; lateralBudIndex++) {
                                 Bud newLateralBud;
                                 newLateralBud.m_status = BudStatus::Sleeping;
                                 newLateralBud.m_newInternodeInfo.m_localRotation = glm::vec3(glm::radians(
                                                                                                      glm::gaussRand(generalTreeParameters.m_branchingAngleMeanVariance.x,
                                                                                                                     generalTreeParameters.m_branchingAngleMeanVariance.y)),

                                                                                              0.0f,
                                                                                              lateralBudIndex *
                                                                                              turnAngle);
                                 internode->m_lateralBuds.push_back(std::move(newLateralBud));
                             }
                         }
                     }
                         break;
                     case BudStatus::Flushed: {
                         for (auto &lateralBud: internode->m_lateralBuds) {
                             if (lateralBud.m_status != BudStatus::Sleeping) continue;
                             if (generalTreeParameters.m_budKillProbabilityApicalLateral.y >
                                 glm::linearRand(0.0f, 1.0f)) {
                                 lateralBud.m_status = BudStatus::Died;
                                 continue;
                             }

                             bool flush = false;
                             float flushProbability = generalTreeParameters.m_lateralBudFlushingProbability * (
                                     internodeIllumination.m_intensity *
                                     generalTreeParameters.m_lateralBudFlushingLightingFactor +
                                     (1.0f - internodeStatus.m_inhibitor)
                             )
                                                      / (generalTreeParameters.m_lateralBudFlushingLightingFactor +
                                                         1.0f);
                             if((internodeInfo.m_neighborsProximity * generalTreeParameters.m_internodeLengthMeanVariance.x) > generalTreeParameters.m_neighborAvoidance.z){
                                 flushProbability = 0;
                             }else{
                                 float avoidance = generalTreeParameters.m_neighborAvoidance.x * glm::pow((internodeInfo.m_neighborsProximity * generalTreeParameters.m_internodeLengthMeanVariance.x), generalTreeParameters.m_neighborAvoidance.y);
                                 flushProbability /= avoidance;
                             }
                             if (flushProbability > glm::linearRand(0.0f, 1.0f)) {
                                 flush = true;
                             }
                             if (flush) {
                                 //Apply tropism to localRotation.
                                 glm::quat desiredGlobalRotation = globalTransform.GetRotation() *
                                                                   lateralBud.m_newInternodeInfo.m_localRotation;
                                 glm::vec3 desiredGlobalFront = desiredGlobalRotation * glm::vec3(0, 0, -1);
                                 glm::vec3 desiredGlobalUp = desiredGlobalRotation * glm::vec3(0, 1, 0);
                                 desiredGlobalUp = glm::rotate(desiredGlobalUp, glm::radians(
                                                                       glm::gaussRand(generalTreeParameters.m_rollAngleMeanVariance.x,
                                                                                      generalTreeParameters.m_rollAngleMeanVariance.y)),
                                                               desiredGlobalFront);
                                 desiredGlobalFront = glm::rotate(desiredGlobalFront, glm::radians(
                                                                          glm::gaussRand(generalTreeParameters.m_apicalAngleMeanVariance.x,
                                                                                         generalTreeParameters.m_apicalAngleMeanVariance.y)),
                                                                  desiredGlobalUp);
                                 ApplyTropism(glm::vec3(0, -1, 0), generalTreeParameters.m_gravitropism,
                                              desiredGlobalFront, desiredGlobalUp);
                                 ApplyTropism(internodeIllumination.m_direction,
                                              generalTreeParameters.m_phototropism, desiredGlobalFront,
                                              desiredGlobalUp);
                                 desiredGlobalRotation = glm::quatLookAt(desiredGlobalFront, desiredGlobalUp);
                                 lateralBud.m_newInternodeInfo = InternodeInfo();
                                 lateralBud.m_newInternodeInfo.m_localRotation =
                                         glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
                                 lateralBud.m_newInternodeInfo.m_thickness = generalTreeParameters.m_endNodeThicknessAndControl.x;
                                 lateralBud.m_newInternodeInfo.m_currentRoot = internodeInfo.m_currentRoot;
                                 lateralBud.m_status = BudStatus::Flushing;
                             }
                         }
                     }
                         break;
                 }
             }, true);
    std::vector<Entity> entities;
    m_internodesQuery.ToEntityArray(entities);
    for (const auto &entity: entities) {
        if (!entity.IsEnabled()) continue;
        auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
        auto parameters = entity.GetDataComponent<GeneralTreeParameters>();
        if (internode->m_apicalBud.m_status == BudStatus::Flushing) {
            auto newInternodeEntity = Retrieve(entity);
            InternodeStatus newInternodeStatus;
            newInternodeStatus.m_desiredLocalRotation = internode->m_apicalBud.m_newInternodeInfo.m_localRotation;
            newInternodeEntity.SetDataComponent(newInternodeStatus);
            newInternodeEntity.SetDataComponent(parameters);
            newInternodeEntity.SetDataComponent(internode->m_apicalBud.m_newInternodeInfo);
            internode->m_apicalBud.m_status = BudStatus::Flushed;
            auto newInternode = newInternodeEntity.GetOrSetPrivateComponent<Internode>().lock();
            newInternode->m_fromApicalBud = true;
        }

        for (auto &bud: internode->m_lateralBuds) {
            if (bud.m_status == BudStatus::Flushing) {
                auto newInternodeEntity = Retrieve(entity);
                InternodeStatus newInternodeStatus;
                newInternodeStatus.m_desiredLocalRotation = bud.m_newInternodeInfo.m_localRotation;
                newInternodeEntity.SetDataComponent(newInternodeStatus);
                newInternodeEntity.SetDataComponent(parameters);
                newInternodeEntity.SetDataComponent(bud.m_newInternodeInfo);
                bud.m_status = BudStatus::Flushed;
                auto newInternode = newInternodeEntity.GetOrSetPrivateComponent<Internode>().lock();
                newInternode->m_fromApicalBud = false;
            }
        }
    }

    EntityManager::ForEach<Transform, InternodeInfo>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity,
                 Transform &transform, InternodeInfo &internodeInfo) {
                 auto parent = entity.GetParent();
                 if (parent.IsNull() || !parent.HasDataComponent<InternodeInfo>()) return;
                 auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                 transform.SetValue(glm::vec3(0.0f, 0.0f, -parentInternodeInfo.m_length),
                                    internodeInfo.m_localRotation, glm::vec3(1.0f));
             }, true);

    ParallelForEachRoot(m_currentPlants, [&](int plantIndex, Entity root) {
        auto parent = root.GetParent();
        auto rootTransform = root.GetDataComponent<Transform>();
        GlobalTransform rootGlobalTransform;
        if (!parent.IsNull()) {
            rootGlobalTransform.m_value =
                    parent.GetDataComponent<GlobalTransform>().m_value * rootTransform.m_value;
        } else {
            rootGlobalTransform.m_value = rootTransform.m_value;
        };
        root.SetDataComponent(rootGlobalTransform);
        TransformManager::CalculateTransformGraphForDescendents(root);
    });
#pragma endregion
#pragma region PostProcess
#pragma region Transform
    ParallelForEachRoot(m_currentPlants, [&](int plantIndex, Entity root) {
        if(!root.GetDataComponent<InternodeInfo>().m_isRealRoot) return;
        TreeGraphWalkerEndToRoot(root, root, [&](Entity parent) {
            float thicknessCollection = 0.0f;
            auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
            auto parameters = parent.GetDataComponent<GeneralTreeParameters>();
            parent.ForEachChild([&](Entity child) {
                if (!InternodeCheck(child)) return;
                auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                thicknessCollection += glm::pow(childInternodeInfo.m_thickness,
                                                1.0f / parameters.m_endNodeThicknessAndControl.y);
            });
            parentInternodeInfo.m_thickness = glm::pow(thicknessCollection,
                                                       parameters.m_endNodeThicknessAndControl.y);
            parent.SetDataComponent(parentInternodeInfo);
        }, [](Entity endNode) {
        });
    });
#pragma endregion

#pragma endregion

}

void GeneralTreeBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled General Tree Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("General Tree Internode", InternodeInfo(), GeneralTreeTag(),
                                                 GeneralTreeParameters(), InternodeStatus(),
                                                 InternodeWaterPressure(), InternodeWater(), InternodeIllumination(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(GeneralTreeTag());
}

void GeneralTreeBehaviour::OnInspect() {
    RecycleButton();
    CreateInternodeMenu<GeneralTreeParameters>
            ("New General Tree Wizard",
             ".gtparams",
             [](GeneralTreeParameters &params) {
                 params.OnInspect();
             },
             [](GeneralTreeParameters &params,
                const std::filesystem::path &path) {
                 params.Load(path);
             },
             [](const GeneralTreeParameters &params,
                const std::filesystem::path &path) {
                 params.Save(path);
             },
             [&](const GeneralTreeParameters &params,
                 const Transform &transform) {
                 return NewPlant(params, transform);
             }
            );

    static float resolution = 0.02f;
    static float subdivision = 4.0f;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate branch mesh")) {
        GenerateSkinnedMeshes(m_internodesQuery, subdivision, resolution);
    }

}

bool GeneralTreeBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<GeneralTreeTag>();
}

Entity GeneralTreeBehaviour::Retrieve() {
    auto retVal = RetrieveHelper<DefaultInternodeResource>();
    retVal.SetDataComponent(InternodeWater());
    retVal.SetDataComponent(InternodeIllumination());
    retVal.SetDataComponent(InternodeStatus());
    return retVal;
}

Entity GeneralTreeBehaviour::Retrieve(const Entity &parent) {
    auto retVal = RetrieveHelper<DefaultInternodeResource>(parent);
    retVal.SetDataComponent(InternodeWater());
    retVal.SetDataComponent(InternodeIllumination());
    retVal.SetDataComponent(InternodeStatus());
    return retVal;
}

Entity GeneralTreeBehaviour::NewPlant(const GeneralTreeParameters &params, const Transform &transform) {
    auto entity = Retrieve();
    Transform internodeTransform;
    internodeTransform.m_value =
            glm::translate(glm::vec3(0.0f)) *
            glm::mat4_cast(glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f))) *
            glm::scale(glm::vec3(1.0f));
    internodeTransform.m_value = transform.m_value * internodeTransform.m_value;
    entity.SetDataComponent(internodeTransform);
    GeneralTreeTag tag;
    entity.SetDataComponent(tag);
    InternodeInfo newInfo;
    newInfo.m_length = 0;
    newInfo.m_isRealRoot = true;
    newInfo.m_thickness = params.m_endNodeThicknessAndControl.x;
    entity.SetDataComponent(newInfo);
    entity.SetDataComponent(params);

    auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
    internode->m_fromApicalBud = true;
    auto waterFeeder = entity.GetOrSetPrivateComponent<InternodeWaterFeeder>().lock();
    return entity;
}

void InternodeWaterFeeder::Clone(const std::shared_ptr<IPrivateComponent> &target) {

}

void InternodeWaterFeeder::OnInspect() {
    ImGui::Text(("Last request:" + std::to_string(m_lastRequest)).c_str());
    ImGui::DragFloat("Water Factor", &m_waterDividends, 0.1f, 0.0f, 9999999.0f);
}

