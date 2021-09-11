//
// Created by lllll on 8/29/2021.
//

#include "GeneralTreeBehaviour.hpp"
#include "InternodeSystem.hpp"
#include <DefaultInternodeResource.hpp>
#include "EmptyInternodeResource.hpp"
#include "TransformManager.hpp"

using namespace PlantArchitect;

void GeneralTreeBehaviour::PreProcess(float deltaTime) {
    m_currentPlants.clear();
    CollectRoots(m_internodesQuery, m_currentPlants);
    int plantSize = m_currentPlants.size();
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
    std::vector<std::vector<float>> totalRequestCollector;
    totalRequestCollector.resize(plantSize);
    for (auto &i: totalRequestCollector) {
        i.resize(JobManager::PrimaryWorkers().Size());
        for (auto &j: i) j = 0;
    }
    EntityManager::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination) {
                 int plantIndex = 0;
                 for (const auto &plant: m_currentPlants) {
                     if (internodeInfo.m_currentRoot == plant) {
                         totalRequestCollector[plantIndex][i] +=
                                 internodeWaterPressure.m_value / internodeStatus.m_distanceToRoot *
                                 internodeIllumination.m_intensity;
                         break;
                     }
                     plantIndex++;
                 }
             }, true);
    std::vector<float> totalRequests;
    totalRequests.resize(plantSize);
    int plantIndex = 0;
    for (const auto &i: totalRequestCollector) {
        for (const auto &j: i) {
            totalRequests[plantIndex] += j;
        }
        plantIndex++;
    }
    std::vector<float> waterDividends;
    waterDividends.resize(plantIndex);
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        waterDividends[plantIndex] = 0;
        if (m_currentPlants[plantIndex].HasPrivateComponent<InternodeWaterFeeder>()) {
            waterDividends[plantIndex] += m_currentPlants[plantIndex].GetOrSetPrivateComponent<InternodeWaterFeeder>().lock()->m_value;
        }
        //Add other water from precipitation, etc.

        waterDividends[plantIndex] /= totalRequests[plantIndex];
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
                                                   internodeStatus.m_distanceToRoot * internodeIllumination.m_intensity;
                         break;
                     }
                     plantIndex++;
                 }
             }, true);
#pragma endregion
}

void GeneralTreeBehaviour::Grow(float deltaTime) {
    std::vector<Entity> entities;
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
                             desiredGlobalUp = glm::rotate(desiredGlobalUp, glm::radians(glm::gaussRand(generalTreeParameters.m_rollAngleMeanVariance.x, generalTreeParameters.m_rollAngleMeanVariance.y)), desiredGlobalFront);
                             desiredGlobalFront = glm::rotate(desiredGlobalFront, glm::radians(glm::gaussRand(generalTreeParameters.m_apicalAngleMeanVariance.x, generalTreeParameters.m_apicalAngleMeanVariance.y)), desiredGlobalUp);
                             ApplyTropism(glm::vec3(0, -1, 0), generalTreeParameters.m_gravitropism, desiredGlobalFront, desiredGlobalUp);
                             ApplyTropism(internodeIllumination.m_direction, generalTreeParameters.m_phototropism, desiredGlobalFront, desiredGlobalUp);
                             internode->m_apicalBud.m_newInternodeInfo.m_localRotation = glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
                             internode->m_apicalBud.m_status = BudStatus::Flushing;
                             //Form lateral buds here.
                             float turnAngle = glm::radians(360.0f / m_lateralBudCount);
                             for(int lateralBudIndex = 0; lateralBudIndex < generalTreeParameters.m_lateralBudCount; i++){
                                 Bud newLateralBud;
                                 newLateralBud.m_status = BudStatus::Sleeping;
                                 newLateralBud.m_newInternodeInfo.m_localRotation = glm::vec3(glm::radians(glm::gaussRand(generalTreeParameters.m_branchingAngleMeanVariance.x, generalTreeParameters.m_branchingAngleMeanVariance.y)), lateralBudIndex * turnAngle, 0.0f);
                                 internode->m_lateralBuds.push_back(std::move(newLateralBud));
                             }
                         }
                     }
                         break;
                     case BudStatus::Flushed: {
                         for (auto &lateralBud: internode->m_lateralBuds) {
                             if (lateralBud.m_status != BudStatus::Sleeping) continue;
                             bool flush = false;
                             float inhibitor = 0.0f;
                             float flushProbability = (
                                     internodeIllumination.m_intensity * generalTreeParameters.m_lateralBudLightingFactor +
                                             (1.0f - inhibitor)
                                                      )
                                     / (generalTreeParameters.m_lateralBudLightingFactor + 1.0f);
                             if(flushProbability > glm::linearRand(0.0f, 1.0f)){
                                 flush = true;
                             }
                             if (flush) {
                                 //Apply tropism to localRotation.
                                 glm::quat desiredGlobalRotation = globalTransform.GetRotation() * lateralBud.m_newInternodeInfo.m_localRotation;
                                 glm::vec3 desiredGlobalFront = desiredGlobalRotation * glm::vec3(0, 0, -1);
                                 glm::vec3 desiredGlobalUp = desiredGlobalRotation * glm::vec3(0, 1, 0);
                                 desiredGlobalUp = glm::rotate(desiredGlobalUp, glm::radians(glm::gaussRand(generalTreeParameters.m_rollAngleMeanVariance.x, generalTreeParameters.m_rollAngleMeanVariance.y)), desiredGlobalFront);
                                 desiredGlobalFront = glm::rotate(desiredGlobalFront, glm::radians(glm::gaussRand(generalTreeParameters.m_apicalAngleMeanVariance.x, generalTreeParameters.m_apicalAngleMeanVariance.y)), desiredGlobalUp);
                                 ApplyTropism(glm::vec3(0, -1, 0), generalTreeParameters.m_gravitropism, desiredGlobalFront, desiredGlobalUp);
                                 ApplyTropism(internodeIllumination.m_direction, generalTreeParameters.m_phototropism, desiredGlobalFront, desiredGlobalUp);
                                 lateralBud.m_newInternodeInfo.m_localRotation = glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
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

    m_internodesQuery.ToEntityArray(entities);
    for (const auto &entity: entities) {
        if (!entity.IsEnabled()) continue;
        auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
        if (internode->m_apicalBud.m_status == BudStatus::Flushing) {
            auto newInternodeEntity = Retrieve(entity);
            newInternodeEntity.SetDataComponent(internode->m_apicalBud.m_newInternodeInfo);
            internode->m_apicalBud.m_status = BudStatus::Flushed;
            auto newInternode = newInternodeEntity.GetOrSetPrivateComponent<Internode>().lock();
            newInternode->m_fromApicalBud = true;
        }

        for (auto &bud: internode->m_lateralBuds) {
            if (bud.m_status == BudStatus::Flushing) {
                auto newInternodeEntity = Retrieve(entity);
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
            rootGlobalTransform.m_value = parent.GetDataComponent<GlobalTransform>().m_value * rootTransform.m_value;
        } else {
            rootGlobalTransform.m_value = rootTransform.m_value;
        };
        root.SetDataComponent(rootGlobalTransform);
        TransformManager::CalculateTransformGraphForDescendents(root);
    });
}

void GeneralTreeBehaviour::PostProcess(float deltaTime) {
    //Calculate local transform.
    EntityManager::ForEach<Transform>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, Transform &transform) {
                 auto parent = entity.GetParent();
                 if (parent.IsNull() || !parent.HasDataComponent<InternodeInfo>()) return;
                 auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                 transform.SetValue(glm::vec3(0, 0, parentInternodeInfo.m_length),
                                    parentInternodeInfo.m_localRotation, glm::vec3(1.0f));
             }, true);
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

    if (ImGui::Button("Create new internode...")) {
        auto entity = Retrieve();
    }
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

void GeneralTreeParameters::OnInspect() {

}

GeneralTreeParameters::GeneralTreeParameters() {

}

void InternodeWaterFeeder::Clone(const std::shared_ptr<IPrivateComponent> &target) {

}
