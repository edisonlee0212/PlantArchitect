//
// Created by lllll on 8/29/2021.
//

#include "GeneralTreeBehaviour.hpp"
#include "InternodeSystem.hpp"
#include <DefaultInternodeResource.hpp>
#include "EmptyInternodeResource.hpp"
#include "TransformManager.hpp"

using namespace PlantArchitect;

void GeneralTreeBehaviour::Grow(int iterations) {
    m_currentPlants.clear();
    CollectRoots(m_internodesQuery, m_currentPlants);
    int plantSize = m_currentPlants.size();
    for (int iteration = 0; iteration < iterations; iteration++) {
#pragma region PreProcess
#pragma region InternodeStatus
        ParallelForEachRoot(m_currentPlants, [&](int plantIndex, Entity root) {
            auto internodeInfo = root.GetDataComponent<InternodeInfo>();
            auto internodeStatus = root.GetDataComponent<InternodeStatus>();
            internodeInfo.m_currentRoot = root;
            internodeStatus.m_distanceToRoot = 0;
            internodeStatus.m_biomass = internodeInfo.m_length * internodeInfo.m_thickness;
            root.SetDataComponent(internodeInfo);
            root.SetDataComponent(internodeStatus);
            TreeGraphWalker(root, root, [&](Entity parent, Entity child) {
                auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
                auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                auto childInternode = child.GetOrSetPrivateComponent<Internode>().lock();
                childInternodeInfo.m_currentRoot = root;
                childInternodeStatus.m_distanceToRoot =
                        parentInternodeInfo.m_length + parentInternodeStatus.m_distanceToRoot;
                childInternodeStatus.m_biomass = childInternodeInfo.m_length * childInternodeInfo.m_thickness;
                child.SetDataComponent(childInternodeInfo);
                child.SetDataComponent(childInternodeStatus);
            }, [&](Entity parent) {
                auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
                parentInternodeStatus.m_totalDistanceToAllBranchEnds = parentInternodeStatus.m_childTotalBiomass = 0;
                float maxDistanceToAnyBranchEnd = -1.0f;
                float maxTotalDistanceToAllBranchEnds = -1.0f;
                float maxChildTotalBiomass = -1.0f;
                parent.ForEachChild([&](Entity child) {
                    auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                    auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                    float childTotalDistanceToAllBranchEnds =
                            childInternodeStatus.m_totalDistanceToAllBranchEnds + childInternodeInfo.m_length;
                    float childTotalBiomass = childInternodeStatus.m_childTotalBiomass + childInternodeStatus.m_biomass;
                    float childMaxDistanceToAnyBranchEnd =
                            childInternodeStatus.m_maxDistanceToAnyBranchEnd + childInternodeInfo.m_length;
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
                parent.SetDataComponent(parentInternodeStatus);
                parent.SetDataComponent(parentInternodeInfo);
            }, [&](Entity endNode) {
                auto endNodeInternodeStatus = endNode.GetDataComponent<InternodeStatus>();
                endNodeInternodeStatus.m_maxDistanceToAnyBranchEnd = endNodeInternodeStatus.m_totalDistanceToAllBranchEnds = endNodeInternodeStatus.m_childTotalBiomass = 0;
                endNodeInternodeStatus.m_largestChild = endNodeInternodeStatus.m_longestChild = endNodeInternodeStatus.m_heaviestChild = Entity();
                endNode.SetDataComponent(endNodeInternodeStatus);
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
        auto *waterFeederEntities = EntityManager::UnsafeGetPrivateComponentOwnersList<InternodeWaterFeeder>();
        if (waterFeederEntities) {
            for (const auto &i: *waterFeederEntities) {
                if(!i.IsEnabled() || !i.HasDataComponent<InternodeWater>()) continue;
                auto waterFeeder = i.GetOrSetPrivateComponent<InternodeWaterFeeder>().lock();
                if(!waterFeeder->IsEnabled()) continue;
                auto internodeWater = i.GetDataComponent<InternodeWater>();
                internodeWater.m_value += waterFeeder->m_waterPerIteration;
                i.SetDataComponent(internodeWater);
            }
        }

        EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination>
                (JobManager::PrimaryWorkers(), m_internodesQuery,
                 [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                     InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination) {
                     auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                     if (internode->m_apicalBud.m_status == BudStatus::Sleeping) {
                         internodeWaterPressure.m_value = 1;
                     } else {
                         internodeWaterPressure.m_value = 0;
                     }
                 }, true);
        std::vector<std::vector<float>> totalWaterCollector;
        std::vector<std::vector<float>> totalRequestCollector;
        totalRequestCollector.resize(JobManager::PrimaryWorkers().Size());
        totalWaterCollector.resize(JobManager::PrimaryWorkers().Size());
        for (auto &i: totalRequestCollector) {
            i.resize(plantSize);
            for (auto &j: i) j = 0;
        }
        for (auto &i: totalWaterCollector) {
            i.resize(plantSize);
            for (auto &j: i) j = 0;
        }
        EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater>
                (JobManager::PrimaryWorkers(), m_internodesQuery,
                 [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                     InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                     InternodeWater &internodeWater) {
                     int plantIndex = 0;
                     for (const auto &plant: m_currentPlants) {
                         if (internodeInfo.m_currentRoot == plant) {
                             totalRequestCollector[i][plantIndex] +=
                                     internodeWaterPressure.m_value / internodeStatus.m_distanceToRoot *
                                     internodeIllumination.m_intensity;
                             totalWaterCollector[i][plantIndex] += internodeWater.m_value;
                             break;
                         }
                         plantIndex++;
                     }
                 }, true);
        std::vector<float> totalRequests;
        std::vector<float> totalWater;
        totalRequests.resize(plantSize);
        totalWater.resize(plantSize);
        for (const auto &i: totalRequestCollector) {
            for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
                totalRequests[plantIndex] += i[plantIndex];
            }
        }
        for (const auto &i: totalWaterCollector) {
            for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
                totalWater[plantIndex] += i[plantIndex];
            }
        }
        std::vector<float> waterDividends;
        waterDividends.resize(plantSize);
        for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
            waterDividends[plantIndex] = totalWater[plantIndex] / totalRequests[plantIndex];
        }
        EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater>
                (JobManager::PrimaryWorkers(), m_internodesQuery,
                 [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                     InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                     InternodeWater &internodeWater) {
                     int plantIndex = 0;
                     for (const auto &plant: m_currentPlants) {
                         if (internodeInfo.m_currentRoot == plant) {
                             internodeWater.m_value += waterDividends[plantIndex] * internodeWaterPressure.m_value /
                                                       internodeStatus.m_distanceToRoot *
                                                       internodeIllumination.m_intensity;
                             break;
                         }
                         plantIndex++;
                     }
                 }, true);
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
                     if (internodeWater.m_value == 0) return;
                     auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                     //1. Internode elongation.
                     switch (internode->m_apicalBud.m_status) {
                         case BudStatus::Sleeping: {
                             if (generalTreeParameters.m_apicalBudKillProbability > glm::linearRand(0.0f, 1.0f)) {
                                 internode->m_apicalBud.m_status = BudStatus::Died;
                                 break;
                             }

                             float desiredLength = glm::gaussRand(generalTreeParameters.m_internodeLengthMeanVariance.x,
                                                                  generalTreeParameters.m_internodeLengthMeanVariance.y);
                             internodeInfo.m_length += internodeWater.m_value;
                             if (internodeInfo.m_length > desiredLength) {
                                 internodeWater.m_value = internodeInfo.m_length - desiredLength;
                                 internodeInfo.m_length = desiredLength;
                                 internode->m_apicalBud.m_newInternodeInfo = InternodeInfo();
                                 internode->m_apicalBud.m_newInternodeInfo.m_thickness = generalTreeParameters.m_endNodeThickness;
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
                                 internode->m_apicalBud.m_newInternodeInfo.m_localRotation =
                                         glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
                                 internode->m_apicalBud.m_status = BudStatus::Flushing;
                                 //Form lateral buds here.
                                 float turnAngle = glm::radians(360.0f / generalTreeParameters.m_lateralBudCount);
                                 for (int lateralBudIndex = 0;
                                      lateralBudIndex < generalTreeParameters.m_lateralBudCount; i++) {
                                     Bud newLateralBud;
                                     newLateralBud.m_status = BudStatus::Sleeping;
                                     newLateralBud.m_newInternodeInfo.m_localRotation = glm::vec3(glm::radians(
                                                                                                          glm::gaussRand(generalTreeParameters.m_branchingAngleMeanVariance.x,
                                                                                                                         generalTreeParameters.m_branchingAngleMeanVariance.y)),
                                                                                                  lateralBudIndex *
                                                                                                  turnAngle, 0.0f);
                                     internode->m_lateralBuds.push_back(std::move(newLateralBud));
                                 }
                             }
                         }
                             break;
                         case BudStatus::Flushed: {
                             for (auto &lateralBud: internode->m_lateralBuds) {
                                 if (lateralBud.m_status != BudStatus::Sleeping) continue;
                                 if (generalTreeParameters.m_lateralBudKillProbability > glm::linearRand(0.0f, 1.0f)) {
                                     lateralBud.m_status = BudStatus::Died;
                                     continue;
                                 }

                                 bool flush = false;
                                 float inhibitor = 0.0f;
                                 float flushProbability = (
                                                                  internodeIllumination.m_intensity *
                                                                  generalTreeParameters.m_lateralBudFlushingLightingFactor +
                                                                  (1.0f - inhibitor)
                                                          )
                                                          / (generalTreeParameters.m_lateralBudFlushingLightingFactor +
                                                             1.0f);
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
                                     lateralBud.m_newInternodeInfo.m_localRotation =
                                             glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
                                     lateralBud.m_newInternodeInfo = InternodeInfo();
                                     lateralBud.m_newInternodeInfo.m_thickness = generalTreeParameters.m_endNodeThickness;
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
            if (internode->m_apicalBud.m_status == BudStatus::Flushing) {
                auto newInternodeEntity = Retrieve(entity);
                newInternodeEntity.SetDataComponent(entity.GetDataComponent<GeneralTreeParameters>());
                newInternodeEntity.SetDataComponent(internode->m_apicalBud.m_newInternodeInfo);
                internode->m_apicalBud.m_status = BudStatus::Flushed;
                auto newInternode = newInternodeEntity.GetOrSetPrivateComponent<Internode>().lock();
                newInternode->m_fromApicalBud = true;
            }

            for (auto &bud: internode->m_lateralBuds) {
                if (bud.m_status == BudStatus::Flushing) {
                    auto newInternodeEntity = Retrieve(entity);
                    newInternodeEntity.SetDataComponent(entity.GetDataComponent<GeneralTreeParameters>());
                    newInternodeEntity.SetDataComponent(bud.m_newInternodeInfo);
                    bud.m_status = BudStatus::Flushed;
                    auto newInternode = newInternodeEntity.GetOrSetPrivateComponent<Internode>().lock();
                    newInternode->m_fromApicalBud = false;
                }
            }
        }

        EntityManager::ForEach<Transform>
                (JobManager::PrimaryWorkers(), m_internodesQuery,
                 [&](int i, Entity entity,
                     Transform &transform) {
                     auto parent = entity.GetParent();
                     if (parent.IsNull() || !parent.HasDataComponent<InternodeInfo>()) return;
                     auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                     transform.SetValue(glm::vec3(0.0f, 0.0f, parentInternodeInfo.m_length),
                                        parentInternodeInfo.m_localRotation, glm::vec3(1.0f));
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
    }
}

void GeneralTreeBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled General Tree Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("General Tree Internode", InternodeInfo(), GeneralTreeTag(),
                                                 GeneralTreeParameters(),
                                                 InternodeWaterPressure(), InternodeWater(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(GeneralTreeTag());
}

void GeneralTreeBehaviour::OnInspect() {
    RecycleButton();
    CreateInternodeMenu<GeneralTreeParameters>
            ("New General Tree Wizard",
             ".scparams",
             [](GeneralTreeParameters &params) {
                 params.OnInspect();
             },
             [](GeneralTreeParameters &params,
                const std::filesystem::path &path) {

             },
             [](const GeneralTreeParameters &params,
                const std::filesystem::path &path) {

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
    return RetrieveHelper<DefaultInternodeResource>();
}

Entity GeneralTreeBehaviour::Retrieve(const Entity &parent) {
    return RetrieveHelper<DefaultInternodeResource>(parent);
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
    newInfo.m_thickness = params.m_endNodeThickness;
    entity.SetDataComponent(newInfo);
    entity.SetDataComponent(params);
    return entity;
}

void GeneralTreeParameters::OnInspect() {
    ImGui::DragInt("Lateral bud per node", &m_lateralBudCount);
    ImGui::DragFloat2("Branching Angle mean/var", &m_branchingAngleMeanVariance.x, 0.01f);
    ImGui::DragFloat2("Roll Angle mean/var", &m_rollAngleMeanVariance.x, 0.01f);
    ImGui::DragFloat2("Apical Angle mean/var", &m_apicalAngleMeanVariance.x, 0.01f);
    ImGui::DragFloat("Gravitropism", &m_gravitropism, 0.01f);
    ImGui::DragFloat("Phototropism", &m_phototropism, 0.01f);
    ImGui::DragFloat2("Internode length mean/var", &m_internodeLengthMeanVariance.x, 0.01f);
    ImGui::DragFloat("End thickness", &m_endNodeThickness, 0.01f);
    ImGui::DragFloat("Lateral bud lighting factor", &m_lateralBudFlushingLightingFactor, 0.01f);
    ImGui::DragFloat("Kill probability apical/lateral", &m_apicalBudKillProbability, 0.01f);
}

GeneralTreeParameters::GeneralTreeParameters() {
    m_lateralBudCount = 2;
    m_branchingAngleMeanVariance = glm::vec2(30, 1);
    m_rollAngleMeanVariance = glm::vec2(30, 1);
    m_apicalAngleMeanVariance = glm::vec2(30, 1);
    m_gravitropism = 0.1f;
    m_phototropism = 0.0f;
    m_internodeLengthMeanVariance = glm::vec2(1, 0.1);
    m_endNodeThickness = 0.01f;

    m_lateralBudFlushingLightingFactor = 0.0;

    m_apicalBudKillProbability = 0;
    m_lateralBudKillProbability = 0;
}

void InternodeWaterFeeder::Clone(const std::shared_ptr<IPrivateComponent> &target) {

}
