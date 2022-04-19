//
// Created by lllll on 8/29/2021.
//

#include "GeneralTreeBehaviour.hpp"
#include "PlantLayer.hpp"
#include <DefaultInternodeResource.hpp>
#include "EmptyInternodeResource.hpp"
#include "TransformLayer.hpp"
#include "DefaultInternodePhyllotaxis.hpp"

using namespace PlantArchitect;

void GeneralTreeBehaviour::Grow(int iteration) {
    std::vector<Entity> currentRoots;
    m_rootsQuery.ToEntityArray(Entities::GetCurrentScene(), currentRoots, true);
    Preprocess(currentRoots);

#pragma region Main Growth
    Entities::ForEach<Transform, GlobalTransform, InternodeInfo, InternodeStatus, InternodeWater, InternodeIllumination>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity,
                 Transform &transform, GlobalTransform &globalTransform,
                 InternodeInfo &internodeInfo, InternodeStatus &internodeStatus,
                 InternodeWater &internodeWater, InternodeIllumination &internodeIllumination) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto rootEntity = internode->m_currentRoot.Get();
                 if (!RootCheck(rootEntity)) return;
                 auto generalTreeParameters = rootEntity.GetOrSetPrivateComponent<Root>().lock()->m_plantDescriptor.Get<GeneralTreeParameters>();
                 internodeStatus.m_age++;
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
                         if (generalTreeParameters->m_budKillProbabilityApicalLateral.x >
                             glm::linearRand(0.0f, 1.0f)) {
                             internode->m_apicalBud.m_status = BudStatus::Died;
                             break;
                         }
                         if (internodeWater.m_value == 0) return;
                         float desiredLength = glm::gaussRand(generalTreeParameters->m_internodeLengthMeanVariance.x,
                                                              generalTreeParameters->m_internodeLengthMeanVariance.y);
                         internodeStatus.m_branchLength -= internodeInfo.m_length;
                         internodeInfo.m_length += internodeWater.m_value;
                         internodeWater.m_value = 0;
                         if (internodeInfo.m_length > desiredLength) {
                             internodeInfo.m_endNode = false;
                             internodeWater.m_value = internodeInfo.m_length - desiredLength;
                             internodeInfo.m_length = desiredLength;
                             internode->m_apicalBud.m_newInternodeInfo = InternodeInfo();
                             internode->m_apicalBud.m_newInternodeInfo.m_layer = internodeInfo.m_layer + 1;
                             internode->m_apicalBud.m_newInternodeInfo.m_thickness = generalTreeParameters->m_endNodeThicknessAndControl.x;
                             glm::quat desiredGlobalRotation = globalTransform.GetRotation();
                             glm::vec3 desiredGlobalFront = desiredGlobalRotation * glm::vec3(0, 0, -1);
                             glm::vec3 desiredGlobalUp = desiredGlobalRotation * glm::vec3(0, 1, 0);
                             desiredGlobalUp = glm::rotate(desiredGlobalUp, glm::radians(
                                                                   glm::gaussRand(generalTreeParameters->m_rollAngleMeanVariance.x,
                                                                                  generalTreeParameters->m_rollAngleMeanVariance.y)),
                                                           desiredGlobalFront);
                             desiredGlobalFront = glm::rotate(desiredGlobalFront, glm::radians(
                                                                      glm::gaussRand(generalTreeParameters->m_apicalAngleMeanVariance.x,
                                                                                     generalTreeParameters->m_apicalAngleMeanVariance.y)),
                                                              desiredGlobalUp);
                             ApplyTropism(glm::vec3(0, -1, 0), generalTreeParameters->m_gravitropism,
                                          desiredGlobalFront, desiredGlobalUp);
                             ApplyTropism(internodeIllumination.m_direction, generalTreeParameters->m_phototropism,
                                          desiredGlobalFront, desiredGlobalUp);

                             desiredGlobalRotation = glm::quatLookAt(desiredGlobalFront, desiredGlobalUp);
                             internode->m_apicalBud.m_newInternodeInfo.m_localRotation =
                                     glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
                             internode->m_apicalBud.m_status = BudStatus::Flushing;
                             //Form lateral buds here.
                             float turnAngle = glm::radians(360.0f / generalTreeParameters->m_lateralBudCount);
                             for (int lateralBudIndex = 0;
                                  lateralBudIndex < generalTreeParameters->m_lateralBudCount; lateralBudIndex++) {
                                 Bud newLateralBud;
                                 newLateralBud.m_status = BudStatus::Sleeping;
                                 newLateralBud.m_newInternodeInfo.m_localRotation = glm::vec3(glm::radians(
                                                                                                      glm::gaussRand(generalTreeParameters->m_branchingAngleMeanVariance.x,
                                                                                                                     generalTreeParameters->m_branchingAngleMeanVariance.y)),

                                                                                              0.0f,
                                                                                              lateralBudIndex *
                                                                                              turnAngle);
                                 internode->m_lateralBuds.push_back(std::move(newLateralBud));
                             }
                         }
                         internodeStatus.m_branchLength += internodeInfo.m_length;
                     }
                         break;
                     case BudStatus::Flushed: {
                         for (auto &lateralBud: internode->m_lateralBuds) {
                             if (lateralBud.m_status != BudStatus::Sleeping) continue;
                             if (generalTreeParameters->m_budKillProbabilityApicalLateral.y >
                                 glm::linearRand(0.0f, 1.0f)) {
                                 lateralBud.m_status = BudStatus::Died;
                                 continue;
                             }

                             bool flush = false;
                             float flushProbability = generalTreeParameters->m_lateralBudFlushingProbability * (
                                     internodeIllumination.m_intensity *
                                     generalTreeParameters->m_lateralBudFlushingLightingFactor +
                                     (1.0f - internodeStatus.m_inhibitor)
                             )
                                                      / (generalTreeParameters->m_lateralBudFlushingLightingFactor +
                                                         1.0f);
                             if ((internodeInfo.m_neighborsProximity *
                                  generalTreeParameters->m_internodeLengthMeanVariance.x) >
                                 generalTreeParameters->m_neighborAvoidance.z) {
                                 flushProbability = 0;
                             } else {
                                 float avoidance = generalTreeParameters->m_neighborAvoidance.x * glm::pow(
                                         (internodeInfo.m_neighborsProximity *
                                          generalTreeParameters->m_internodeLengthMeanVariance.x),
                                         generalTreeParameters->m_neighborAvoidance.y);
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
                                                                       glm::gaussRand(generalTreeParameters->m_rollAngleMeanVariance.x,
                                                                                      generalTreeParameters->m_rollAngleMeanVariance.y)),
                                                               desiredGlobalFront);
                                 desiredGlobalFront = glm::rotate(desiredGlobalFront, glm::radians(
                                                                          glm::gaussRand(generalTreeParameters->m_apicalAngleMeanVariance.x,
                                                                                         generalTreeParameters->m_apicalAngleMeanVariance.y)),
                                                                  desiredGlobalUp);
                                 ApplyTropism(glm::vec3(0, -1, 0), generalTreeParameters->m_gravitropism,
                                              desiredGlobalFront, desiredGlobalUp);
                                 ApplyTropism(internodeIllumination.m_direction,
                                              generalTreeParameters->m_phototropism, desiredGlobalFront,
                                              desiredGlobalUp);
                                 lateralBud.m_flushProbability = flushProbability;
                                 lateralBud.m_newInternodeInfo = InternodeInfo();
                                 lateralBud.m_newInternodeInfo.m_layer = internodeInfo.m_layer + 1;
                                 lateralBud.m_newInternodeInfo.m_localRotation =
                                         glm::inverse(globalTransform.GetRotation()) * desiredGlobalRotation;
                                 lateralBud.m_newInternodeInfo.m_thickness = generalTreeParameters->m_endNodeThicknessAndControl.x;
                                 lateralBud.m_status = BudStatus::Flushing;
                             }
                         }
                     }
                         break;
                 }
             }, true);
    std::vector<Entity> entities;
    m_internodesQuery.ToEntityArray(Entities::GetCurrentScene(), entities);
    int currentNodeCount = entities.size();

    for (const auto &entity: entities) {
        if (!entity.IsEnabled()) continue;
        auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
        auto internodeInfo = entity.GetDataComponent<InternodeInfo>();
        if (internode->m_apicalBud.m_status == BudStatus::Flushing) {
            auto internodeStatus = entity.GetDataComponent<InternodeStatus>();
            auto newInternodeEntity = CreateInternode(entity);
            InternodeStatus newInternodeStatus;
            newInternodeStatus.m_desiredLocalRotation = internode->m_apicalBud.m_newInternodeInfo.m_localRotation;
            newInternodeStatus.m_branchingOrder = 0;
            newInternodeStatus.m_recordedProbability = internode->m_apicalBud.m_flushProbability;
            newInternodeStatus.m_branchLength =
                    internodeStatus.m_branchLength + internode->m_apicalBud.m_newInternodeInfo.m_length;
            newInternodeStatus.m_currentTotalNodeCount = currentNodeCount;
            newInternodeStatus.m_startDensity = internodeInfo.m_neighborsProximity;
            newInternodeEntity.SetDataComponent(newInternodeStatus);
            newInternodeEntity.SetDataComponent(internode->m_apicalBud.m_newInternodeInfo);
            internode->m_apicalBud.m_status = BudStatus::Flushed;
            auto newInternode = newInternodeEntity.GetOrSetPrivateComponent<Internode>().lock();
            newInternode->m_fromApicalBud = true;
        }
        int branchingOrder = 1;
        for (auto &bud: internode->m_lateralBuds) {
            if (bud.m_status == BudStatus::Flushing) {
                auto newInternodeEntity = CreateInternode(entity);
                InternodeStatus newInternodeStatus;
                newInternodeStatus.m_branchingOrder = branchingOrder;
                newInternodeStatus.m_desiredLocalRotation = bud.m_newInternodeInfo.m_localRotation;
                newInternodeStatus.m_branchLength = bud.m_newInternodeInfo.m_length;
                newInternodeStatus.m_recordedProbability = bud.m_flushProbability;
                newInternodeStatus.m_currentTotalNodeCount = currentNodeCount;
                newInternodeStatus.m_startDensity = internodeInfo.m_neighborsProximity;
                newInternodeEntity.SetDataComponent(newInternodeStatus);
                newInternodeEntity.SetDataComponent(bud.m_newInternodeInfo);
                bud.m_status = BudStatus::Flushed;
                auto newInternode = newInternodeEntity.GetOrSetPrivateComponent<Internode>().lock();
                newInternode->m_fromApicalBud = false;
            }
            branchingOrder++;
        }
    }

    Entities::ForEach<Transform, InternodeInfo>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity,
                 Transform &transform, InternodeInfo &internodeInfo) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto parent = entity.GetParent();
                 if (parent.IsNull() || !parent.HasDataComponent<InternodeInfo>()) return;
                 auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                 transform.SetValue(glm::vec3(0.0f, 0.0f, -parentInternodeInfo.m_length),
                                    internodeInfo.m_localRotation, glm::vec3(1.0f));
             }, true);

    Entities::ForEach<Transform, GlobalTransform>(Entities::GetCurrentScene(), Jobs::Workers(), m_rootsQuery,
                                                  [&](int i, Entity entity,
                                                      Transform &transform, GlobalTransform &globalTransform) {
                                                      Application::GetLayer<TransformLayer>()->CalculateTransformGraphForDescendents(
                                                              Entities::GetCurrentScene(),
                                                              entity);
                                                  }, true);
    /*
    ParallelForEachRoot(m_currentRoots, [&](int plantIndex, Entity root) {
        auto parent = root.GetParent();
        auto rootTransform = root.GetDataComponent<Transform>();
        GlobalTransform rootGlobalTransform;
        if (!parent.IsNull()) {
            rootGlobalTransform.m_value =
                    parent.GetDataComponent<GlobalTransform>().m_value * rootTransform.m_value;
        } else {
            rootGlobalTransform.m_value = rootTransform.m_value;
        }
        root.SetDataComponent(rootGlobalTransform);

    });
     */
#pragma endregion
#pragma region PostProcess
#pragma region Transform
    Entities::ForEach<Transform, GlobalTransform>(Entities::GetCurrentScene(), Jobs::Workers(),
                                                                         m_rootsQuery,
                                                                         [&](int i, Entity entity,
                                                                             Transform &transform,
                                                                             GlobalTransform &globalTransform) {
                                                                             entity.ForEachChild(
                                                                                     [&](const std::shared_ptr<Scene> &currentScene,
                                                                                         Entity child) {
                                                                                         if (!InternodeCheck(
                                                                                                 child))
                                                                                             return;
                                                                                         auto generalTreeParameters = entity.GetOrSetPrivateComponent<Root>().lock()->m_plantDescriptor.Get<GeneralTreeParameters>();
                                                                                         InternodeGraphWalkerEndToRoot(
                                                                                                 child,
                                                                                                 [&](Entity parent) {
                                                                                                     float thicknessCollection = 0.0f;
                                                                                                     auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                                                                                                     parent.ForEachChild(
                                                                                                             [&](const std::shared_ptr<Scene> &scene,
                                                                                                                 Entity child) {
                                                                                                                 if (!InternodeCheck(
                                                                                                                         child))
                                                                                                                     return;
                                                                                                                 auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                                                                                 thicknessCollection += glm::pow(
                                                                                                                         childInternodeInfo.m_thickness,
                                                                                                                         1.0f /
                                                                                                                         generalTreeParameters->m_endNodeThicknessAndControl.y);
                                                                                                             });
                                                                                                     parentInternodeInfo.m_thickness = glm::pow(
                                                                                                             thicknessCollection,
                                                                                                             generalTreeParameters->m_endNodeThicknessAndControl.y);
                                                                                                     parent.SetDataComponent(
                                                                                                             parentInternodeInfo);
                                                                                                 },
                                                                                                 [](Entity endNode) {
                                                                                                 });
                                                                                         CalculateChainDistance(child,
                                                                                                                0);
                                                                                     });
                                                                         }, true);
#pragma endregion


#pragma endregion
    UpdateBranches();
}

void GeneralTreeBehaviour::CalculateChainDistance(const Entity &target, float previousChainDistance) {
    int childAmount = 0;
    target.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (!InternodeCheck(child)) return;
        childAmount += 1;
    });
    auto internodeInfo = target.GetDataComponent<InternodeInfo>();
    auto internodeStatus = target.GetDataComponent<InternodeStatus>();
    internodeStatus.m_chainDistance = previousChainDistance + internodeInfo.m_length;
    target.SetDataComponent(internodeStatus);
    if (childAmount == 1) {
        target.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!InternodeCheck(child)) return;
            CalculateChainDistance(child, internodeStatus.m_chainDistance);
        });
    } else if (childAmount > 1) {
        target.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!InternodeCheck(child)) return;
            CalculateChainDistance(child, 0);
        });
    }
}

GeneralTreeBehaviour::GeneralTreeBehaviour() {
    m_typeName = "GeneralTreeBehaviour";
    m_internodeArchetype =
            Entities::CreateEntityArchetype("General Tree Internode", InternodeInfo(), InternodeStatistics(),
                                            GeneralTreeTag(),
                                            InternodeStatus(),
                                            InternodeWaterPressure(), InternodeWater(), InternodeIllumination(),
                                            InternodeColor(), InternodeCylinder(), InternodeCylinderWidth(),
                                            InternodePointer());
    m_internodesQuery = Entities::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo(), GeneralTreeTag());

    m_rootArchetype =
            Entities::CreateEntityArchetype("General Tree Root", RootInfo(),
                                            GeneralTreeTag());
    m_rootsQuery = Entities::CreateEntityQuery();
    m_rootsQuery.SetAllFilters(RootInfo(), GeneralTreeTag());

    m_branchArchetype =
            Entities::CreateEntityArchetype("General Tree Branch", BranchInfo(),
                                            GeneralTreeTag(),
                                            BranchColor(), BranchCylinder(), BranchCylinderWidth());
    m_branchesQuery = Entities::CreateEntityQuery();
    m_branchesQuery.SetAllFilters(BranchInfo(), GeneralTreeTag());
}

void GeneralTreeBehaviour::OnInspect() {
    FileUtils::OpenFile("Import Graph", "YAML", {".yml"}, [&](const std::filesystem::path &path) {
        auto parameters = ProjectManager::CreateTemporaryAsset<GeneralTreeParameters>();
        ImportGraphTree(path, parameters);
    }, false);
}

bool GeneralTreeBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<GeneralTreeTag>();
}

bool GeneralTreeBehaviour::InternalRootCheck(const Entity &target) {
    return target.HasDataComponent<GeneralTreeTag>();
}

bool GeneralTreeBehaviour::InternalBranchCheck(const Entity &target) {
    return target.HasDataComponent<GeneralTreeTag>();
}

Entity GeneralTreeBehaviour::CreateRoot(AssetRef descriptor, Entity &rootInternode, Entity &rootBranch) {
    auto root = CreateRootHelper<DefaultInternodeResource>(descriptor, rootInternode, rootBranch);
    rootInternode.SetDataComponent(InternodeWater());
    rootInternode.SetDataComponent(InternodeIllumination());
    rootInternode.SetDataComponent(InternodeStatus());
    return root;
}

Entity GeneralTreeBehaviour::CreateInternode(const Entity &parent) {
    auto retVal = CreateInternodeHelper<DefaultInternodeResource>(parent);
    retVal.SetDataComponent(InternodeWater());
    retVal.SetDataComponent(InternodeIllumination());
    retVal.SetDataComponent(InternodeStatus());
    return retVal;
}


Entity GeneralTreeBehaviour::NewPlant(const std::shared_ptr<GeneralTreeParameters> &descriptor, const Transform &transform) {
    Entity rootInternode, rootBranch;
    auto rootEntity = CreateRoot(descriptor, rootInternode, rootBranch);
    auto root = rootEntity.GetOrSetPrivateComponent<Root>().lock();
    root->m_foliagePhyllotaxis = ProjectManager::CreateTemporaryAsset<DefaultInternodePhyllotaxis>();

    Transform internodeTransform;
    internodeTransform.m_value =
            glm::translate(glm::vec3(0.0f)) *
            glm::mat4_cast(glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f))) *
            glm::scale(glm::vec3(1.0f));
    internodeTransform.m_value = transform.m_value * internodeTransform.m_value;
    rootInternode.SetDataComponent(internodeTransform);
    GeneralTreeTag tag;
    rootInternode.SetDataComponent(tag);
    InternodeInfo newInfo;
    newInfo.m_length = 0;
    newInfo.m_layer = 0;
    newInfo.m_thickness = descriptor->m_endNodeThicknessAndControl.x;
    rootInternode.SetDataComponent(newInfo);

    auto internode = rootInternode.GetOrSetPrivateComponent<Internode>().lock();
    internode->m_fromApicalBud = true;

    auto waterFeeder = rootInternode.GetOrSetPrivateComponent<InternodeWaterFeeder>().lock();

    auto branch = rootBranch.GetOrSetPrivateComponent<Branch>().lock();

    return rootEntity;
}

Entity
GeneralTreeBehaviour::ImportGraphTree(const std::filesystem::path &path, AssetRef descriptor) {
    Entity root = Entity();
    if (!std::filesystem::exists(path)) {
        UNIENGINE_ERROR("Not exist!");
        return root;
    }
    try {
        std::ifstream stream(path.string());
        std::stringstream stringStream;
        stringStream << stream.rdbuf();
        YAML::Node in = YAML::Load(stringStream.str());
        auto name = in["name"].as<std::string>();
        int layerSize = in["layersize"].as<int>();
        auto layers = in["layers"];
        auto rootLayer = layers["0"];
        Entity rootInternode, rootBranch;
        auto root = CreateRoot(descriptor, rootInternode, rootBranch);
        root.SetName(name);
        std::unordered_map<int, GanNode> previousNodes;
        GanNode rootNode;
        rootNode.m_start = glm::vec3(0, 0, 0);
        rootNode.m_globalRotation = glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
        rootNode.m_internode = rootInternode;
        int rootIndex = 0;
        rootNode.m_length = 1.0f;
        InternodeInfo rootInternodeInfo;
        rootInternodeInfo.m_thickness = rootLayer["0"]["thickness"].as<float>();
        rootInternodeInfo.m_length = rootNode.m_length;
        GlobalTransform rootGlobalTransform;
        rootGlobalTransform.SetRotation(rootNode.m_globalRotation);
        rootGlobalTransform.SetPosition(rootNode.m_start);
        rootInternode.SetDataComponent(rootGlobalTransform);
        rootInternode.SetDataComponent(rootInternodeInfo);
        previousNodes[0] = rootNode;
        for (int layerIndex = 0; layerIndex < layerSize; layerIndex++) {
            auto layer = layers[std::to_string(layerIndex)];
            auto internodeSize = layer["internodesize"].as<int>();
            for (int nodeIndex = 0; nodeIndex < internodeSize; nodeIndex++) {
                auto node = layer[std::to_string(nodeIndex)];
                auto id = node["id"].as<int>();
                auto parentNodeId = node["parent"].as<int>();
                if (parentNodeId == -1) parentNodeId = 0;
                auto &parentGanNode = previousNodes[parentNodeId];
                GanNode ganNode;
                ganNode.m_internode = CreateInternode(parentGanNode.m_internode);
                ganNode.m_start = parentGanNode.m_start + parentGanNode.m_length *
                                                          (glm::normalize(parentGanNode.m_globalRotation) *
                                                           glm::vec3(0, 0, -1));
#pragma region GlobalTransform
                ganNode.m_length = node["length"].as<float>();
                int index = 0;
                for (const auto &component: node["quat"]) {
                    ganNode.m_globalRotation[index] = component.as<float>();
                    index++;
                }
                GlobalTransform globalTransform;
                globalTransform.SetRotation(ganNode.m_globalRotation);
                globalTransform.SetPosition(ganNode.m_start);
                ganNode.m_internode.SetDataComponent(globalTransform);
#pragma endregion

                InternodeStatus internodeStatus;
                if (node["age"]) {
                    int age = node["age"].as<int>();
                    internodeStatus.m_age = age;
                }
                if (node["childorder"]) {
                    int childorder = node["childorder"].as<int>();

                }
                if (node["level"]) {
                    int level = node["level"].as<int>();
                    internodeStatus.m_level = level;
                }
                ganNode.m_internode.SetDataComponent(internodeStatus);
                InternodeInfo internodeInfo;
                internodeInfo.m_thickness = node["thickness"].as<float>();
                internodeInfo.m_length = ganNode.m_length;
                ganNode.m_internode.SetDataComponent(internodeInfo);
                previousNodes[id] = ganNode;
            }
        }

        Entities::ForEach<GlobalTransform, Transform, InternodeInfo, InternodeStatus>
                (Entities::GetCurrentScene(),
                 Jobs::Workers(), m_internodesQuery,
                 [](int i, Entity entity,
                    GlobalTransform &globalTransform,
                    Transform &transform,
                    InternodeInfo &internodeInfo, InternodeStatus &internodeStatus) {
                     auto parent = entity.GetParent();
                     if (parent.IsNull()) {
                         internodeInfo.m_localRotation = globalTransform.GetRotation();
                     } else {
                         auto parentGlobalTransform = parent.GetDataComponent<GlobalTransform>();
                         auto parentGlobalRotation = parentGlobalTransform.GetRotation();
                         internodeInfo.m_localRotation =
                                 glm::inverse(parentGlobalRotation) * globalTransform.GetRotation();
                     }
                     auto childAmount = entity.GetChildrenAmount();
                     internodeInfo.m_endNode = childAmount == 0;
                     auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 }
                );
    }
    catch (std::exception e) {
        UNIENGINE_ERROR("Failed to load!");
    }
    return root;
}

void GeneralTreeBehaviour::Preprocess(std::vector<Entity> &currentRoots) {
    auto plantSize = currentRoots.size();
#pragma region PreProcess
#pragma region InternodeStatus
    Entities::ForEach<Transform>
            (Entities::GetCurrentScene(),
             Jobs::Workers(), m_rootsQuery,
             [&](int i, Entity entity, Transform &transform
             ) {
                 if (!RootCheck(entity)) return;
                 auto root = entity.GetOrSetPrivateComponent<Root>().lock();
                 auto generalTreeParameters = root->m_plantDescriptor.Get<GeneralTreeParameters>();
                 int amount = 1;
                 auto center = glm::vec3(0);
                 entity.ForEachChild(
                         [&](const std::shared_ptr<Scene> &currentScene,
                             Entity child) {
                             if (!InternodeCheck(child)) return;
                             auto internodeInfo = child.GetDataComponent<InternodeInfo>();
                             auto internodeStatus = child.GetDataComponent<InternodeStatus>();
                             auto internodeGlobalTransform = child.GetDataComponent<GlobalTransform>();
                             auto internode = child.GetOrSetPrivateComponent<Internode>().lock();
                             internodeStatus.m_rootDistance = 0;
                             internodeInfo.m_endNode = false;
                             internodeStatus.m_biomass =
                                     internodeInfo.m_length * internodeInfo.m_thickness * internodeInfo.m_thickness;
                             child.SetDataComponent(internodeInfo);
                             child.SetDataComponent(internodeStatus);
                             center += internodeGlobalTransform.GetPosition();
                             InternodeGraphWalker(child,
                                                  [&](Entity parent, Entity child) {
                                                      auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                                                      auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
                                                      auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                      auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                                                      auto childGlobalTransform = child.GetDataComponent<GlobalTransform>();
                                                      center += childGlobalTransform.GetPosition();
                                                      amount++;
                                                      //Low branch pruning.
                                                      if (childInternodeStatus.m_order != 0) {
                                                          if (childInternodeStatus.m_rootDistance /
                                                              internodeStatus.m_maxDistanceToAnyBranchEnd <
                                                              generalTreeParameters->m_lowBranchPruning) {
                                                              DestroyInternode(child);
                                                              return;
                                                          }
                                                      }
                                                      if (childGlobalTransform.GetPosition().y < 0) {
                                                          DestroyInternode(child);
                                                          return;
                                                      }
                                                      auto childInternode = child.GetOrSetPrivateComponent<Internode>().lock();
                                                      childInternodeStatus.m_rootDistance =
                                                              parentInternodeInfo.m_length +
                                                              parentInternodeStatus.m_rootDistance;
                                                      childInternodeStatus.m_biomass =
                                                              childInternodeInfo.m_length *
                                                              childInternodeInfo.m_thickness;
                                                      if (!childInternode->m_fromApicalBud) {
                                                          childInternodeStatus.m_order =
                                                                  parentInternodeStatus.m_order + 1;
                                                      } else {
                                                          childInternodeStatus.m_order = parentInternodeStatus.m_order;
                                                      }
                                                      child.SetDataComponent(childInternodeInfo);
                                                      child.SetDataComponent(childInternodeStatus);
                                                  },
                                                  [&](Entity parent) {
                                                      auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                                                      auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
                                                      parentInternodeInfo.m_endNode = false;
                                                      parentInternodeStatus.m_inhibitor = 0;
                                                      parentInternodeStatus.m_totalDistanceToAllBranchEnds = parentInternodeStatus.m_childTotalBiomass = 0;
                                                      float maxDistanceToAnyBranchEnd = -1.0f;
                                                      float maxTotalDistanceToAllBranchEnds = -1.0f;
                                                      float maxChildTotalBiomass = -1.0f;
                                                      Entity largestChild;
                                                      Entity longestChild;
                                                      Entity heaviestChild;
                                                      parent.ForEachChild(
                                                              [&](const std::shared_ptr<Scene> &scene, Entity child) {
                                                                  if (!InternodeCheck(child)) return;
                                                                  auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                                  auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                                                                  if (childInternodeInfo.m_endNode) {
                                                                      auto endNodeInternode = child.GetOrSetPrivateComponent<Internode>().lock();
                                                                      float randomFactor = glm::min(
                                                                              generalTreeParameters->m_randomPruningBaseAgeMax.z,
                                                                              generalTreeParameters->m_randomPruningBaseAgeMax.x +
                                                                              generalTreeParameters->m_randomPruningBaseAgeMax.y *
                                                                              childInternodeStatus.m_age);
                                                                      if (childInternodeStatus.m_order >
                                                                          generalTreeParameters->m_randomPruningOrderProtection &&
                                                                          randomFactor > glm::linearRand(0.0f, 1.0f)) {
                                                                          DestroyInternode(child);
                                                                          return;
                                                                      }
                                                                      parentInternodeStatus.m_inhibitor +=
                                                                              generalTreeParameters->m_apicalDominanceBaseAgeDist.x *
                                                                              glm::pow(
                                                                                      generalTreeParameters->m_apicalDominanceBaseAgeDist.y,
                                                                                      childInternodeStatus.m_age);
                                                                  } else {
                                                                      parentInternodeStatus.m_inhibitor +=
                                                                              childInternodeStatus.m_inhibitor *
                                                                              glm::pow(
                                                                                      generalTreeParameters->m_apicalDominanceBaseAgeDist.z,
                                                                                      parentInternodeInfo.m_length);
                                                                  }

                                                                  float childTotalDistanceToAllBranchEnds =
                                                                          childInternodeStatus.m_totalDistanceToAllBranchEnds +
                                                                          childInternodeInfo.m_length;
                                                                  float childTotalBiomass =
                                                                          childInternodeStatus.m_childTotalBiomass +
                                                                          childInternodeStatus.m_biomass;
                                                                  float childMaxDistanceToAnyBranchEnd =
                                                                          childInternodeStatus.m_maxDistanceToAnyBranchEnd +
                                                                          childInternodeInfo.m_length;
                                                                  parentInternodeStatus.m_totalDistanceToAllBranchEnds += childTotalDistanceToAllBranchEnds;
                                                                  parentInternodeStatus.m_childTotalBiomass += childTotalBiomass;
                                                                  if (maxTotalDistanceToAllBranchEnds <
                                                                      childTotalDistanceToAllBranchEnds) {
                                                                      maxTotalDistanceToAllBranchEnds = childTotalDistanceToAllBranchEnds;
                                                                      largestChild = child;
                                                                  }
                                                                  if (maxDistanceToAnyBranchEnd <
                                                                      childMaxDistanceToAnyBranchEnd) {
                                                                      maxDistanceToAnyBranchEnd = childMaxDistanceToAnyBranchEnd;
                                                                      longestChild = child;
                                                                  }
                                                                  if (maxChildTotalBiomass < childTotalBiomass) {
                                                                      maxChildTotalBiomass = childTotalBiomass;
                                                                      heaviestChild = child;
                                                                  }
                                                              });
                                                      parent.ForEachChild(
                                                              [&](const std::shared_ptr<Scene> &scene, Entity child) {
                                                                  if (!InternodeCheck(child)) return;
                                                                  auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                                                                  childInternodeStatus.m_largestChild =
                                                                          largestChild == child;
                                                                  childInternodeStatus.m_longestChild =
                                                                          longestChild == child;
                                                                  childInternodeStatus.m_heaviestChild =
                                                                          heaviestChild == child;
                                                                  child.SetDataComponent(childInternodeStatus);
                                                              });
                                                      parentInternodeStatus.m_maxDistanceToAnyBranchEnd = maxDistanceToAnyBranchEnd;
                                                      parentInternodeStatus.m_sagging =
                                                              glm::min(
                                                                      generalTreeParameters->m_saggingFactorThicknessReductionMax.z,
                                                                      generalTreeParameters->m_saggingFactorThicknessReductionMax.x *
                                                                      parentInternodeStatus.m_childTotalBiomass /
                                                                      glm::pow(parentInternodeInfo.m_thickness /
                                                                               generalTreeParameters->m_endNodeThicknessAndControl.x,
                                                                               generalTreeParameters->m_saggingFactorThicknessReductionMax.y));
                                                      parent.SetDataComponent(parentInternodeStatus);
                                                      parent.SetDataComponent(parentInternodeInfo);
                                                  },
                                                  [&](Entity endNode) {
                                                      auto endNodeInternodeInfo = endNode.GetDataComponent<InternodeInfo>();
                                                      auto endNodeInternodeStatus = endNode.GetDataComponent<InternodeStatus>();
                                                      endNodeInternodeInfo.m_endNode = true;
                                                      endNodeInternodeStatus.m_inhibitor = 0.0f;
                                                      endNodeInternodeStatus.m_maxDistanceToAnyBranchEnd = endNodeInternodeStatus.m_totalDistanceToAllBranchEnds = endNodeInternodeStatus.m_childTotalBiomass = 0;
                                                      endNodeInternodeStatus.m_largestChild = endNodeInternodeStatus.m_longestChild = endNodeInternodeStatus.m_heaviestChild = true;
                                                      endNode.SetDataComponent(endNodeInternodeInfo);
                                                      endNode.SetDataComponent(endNodeInternodeStatus);
                                                  });

                         });
                 root->m_center = center / static_cast<float>(amount);
             });
    Entities::ForEach<Transform>
            (Entities::GetCurrentScene(),
             Jobs::Workers(), m_rootsQuery,
             [&](int i, Entity entity, Transform &transform
             ) {
                 entity.ForEachChild(
                         [&](const std::shared_ptr<Scene> &currentScene,
                             Entity child) {
                             if (!InternodeCheck(child)) return;
                             InternodeGraphWalkerRootToEnd(child, [](Entity parent, Entity child) {
                                 auto parentInternodeStatus = parent.GetDataComponent<InternodeStatus>();
                                 auto childInternodeStatus = child.GetDataComponent<InternodeStatus>();
                                 if (childInternodeStatus.m_largestChild) {
                                     childInternodeStatus.m_level = parentInternodeStatus.m_level;
                                 } else {
                                     childInternodeStatus.m_level = parentInternodeStatus.m_level + 1;
                                 }
                                 child.SetDataComponent(childInternodeStatus);
                             });
                         });
             });
#pragma endregion
    auto workerSize = Jobs::Workers().Size();
#pragma region Illumination
    Entities::ForEach<InternodeInfo, InternodeIllumination, GlobalTransform>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeTag,
                 InternodeIllumination &internodeIllumination,
                 GlobalTransform &internodeGlobalTransform) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto rootEntity = internode->m_currentRoot.Get();
                 if (!RootCheck(rootEntity)) return;
                 auto root = rootEntity.GetOrSetPrivateComponent<Root>().lock();
                 auto difference =
                         internodeGlobalTransform.GetPosition() - root->m_center;
                 internodeIllumination.m_direction = glm::normalize(difference);
                 internodeIllumination.m_intensity = 1.0f;
                 if (internodeIllumination.m_intensity < 0.1) {
                     internodeIllumination.m_direction = glm::vec3(0.0f, 1.0f, 0.0f);
                     internodeIllumination.m_intensity = 0.1f;
                 }
             }, true);
#pragma endregion
#pragma region Water
#pragma region Collect water requests
    Entities::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo,
                 InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 if (internode->m_apicalBud.m_status == BudStatus::Sleeping) {
                     internodeWaterPressure.m_value = 1;
                 } else {
                     internodeWaterPressure.m_value = 0;
                 }
             }, true);

    std::vector<std::vector<float>> totalRequestCollector;
    totalRequestCollector.resize(workerSize);
    for (auto &i: totalRequestCollector) {
        i.resize(plantSize);
        for (auto &j: i) j = 0;
    }
    Entities::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo,
                 InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                 InternodeWater &internodeWater) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto rootEntity = internode->m_currentRoot.Get();
                 if (!RootCheck(rootEntity)) return;
                 auto generalTreeParameters = rootEntity.GetOrSetPrivateComponent<Root>().lock()->m_plantDescriptor.Get<GeneralTreeParameters>();
                 int plantIndex = 0;
                 for (const auto &plant: currentRoots) {
                     if (rootEntity == plant) {
                         internodeStatus.CalculateApicalControl(generalTreeParameters->m_apicalControl);
                         totalRequestCollector[i % workerSize][plantIndex] +=
                                 internodeStatus.m_apicalControl *
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
    Entities::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo,
                 InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                 InternodeWater &internodeWater) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto rootEntity = internode->m_currentRoot.Get();
                 if (!RootCheck(rootEntity)) return;
                 int plantIndex = 0;
                 for (const auto &plant: currentRoots) {
                     if (rootEntity == plant) {
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
    Entities::ForEach<InternodeInfo, InternodeWaterPressure, InternodeStatus, InternodeIllumination, InternodeWater>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo,
                 InternodeWaterPressure &internodeWaterPressure,
                 InternodeStatus &internodeStatus, InternodeIllumination &internodeIllumination,
                 InternodeWater &internodeWater) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto rootEntity = internode->m_currentRoot.Get();
                 if (!RootCheck(rootEntity)) return;
                 int plantIndex = 0;
                 for (const auto &plant: currentRoots) {
                     if (rootEntity == plant) {
                         internodeWater.m_value =
                                 internodeStatus.m_apicalControl *
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
}

Entity GeneralTreeBehaviour::CreateBranch(const Entity &parent, const Entity &internode) {
    auto retVal = CreateBranchHelper(parent, internode);
    return retVal;
}


void InternodeWaterFeeder::OnInspect() {
    ImGui::Text(("Last request:" + std::to_string(m_lastRequest)).c_str());
    ImGui::DragFloat("Water Factor", &m_waterDividends, 0.1f, 0.0f, 9999999.0f);
}


void GeneralTreeParameters::OnInspect() {
    IPlantDescriptor::OnInspect();
    if(ImGui::TreeNodeEx("Structure", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragInt("Lateral bud per node", &m_lateralBudCount);
        ImGui::DragFloat2("Branching Angle mean/var", &m_branchingAngleMeanVariance.x, 0.01f);
        ImGui::DragFloat2("Roll Angle mean/var", &m_rollAngleMeanVariance.x, 0.01f);
        ImGui::DragFloat2("Apical Angle mean/var", &m_apicalAngleMeanVariance.x, 0.01f);
        ImGui::DragFloat("Gravitropism", &m_gravitropism, 0.01f);
        ImGui::DragFloat("Phototropism", &m_phototropism, 0.01f);
        ImGui::DragFloat2("Internode length mean/var", &m_internodeLengthMeanVariance.x, 0.01f);
        ImGui::DragFloat2("Thickness min/factor", &m_endNodeThicknessAndControl.x, 0.01f);
        ImGui::TreePop();
    }
    if(ImGui::TreeNodeEx("Bud", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat("Lateral bud flushing probability", &m_lateralBudFlushingProbability, 0.01f);
        ImGui::DragFloat3("Neighbor avoidance mul/factor/max", &m_neighborAvoidance.x, 0.001f);
        ImGui::DragFloat("Apical control", &m_apicalControl, 0.01f);
        ImGui::DragFloat3("Apical dominance base/age/dist", &m_apicalDominanceBaseAgeDist.x, 0.01f);
        int maxAgeBeforeInhibitorEnds = m_apicalDominanceBaseAgeDist.x / m_apicalDominanceBaseAgeDist.y;
        float maxDistance = m_apicalDominanceBaseAgeDist.x / m_apicalDominanceBaseAgeDist.z;
        ImGui::Text("Max age / distance: [%i, %.3f]", maxAgeBeforeInhibitorEnds, maxDistance);

        ImGui::DragFloat("Lateral bud lighting factor", &m_lateralBudFlushingLightingFactor, 0.01f);
        ImGui::DragFloat2("Kill probability apical/lateral", &m_budKillProbabilityApicalLateral.x, 0.01f);
        ImGui::TreePop();
    }
    if(ImGui::TreeNodeEx("Internode")) {
        ImGui::DragInt("Random pruning Order Protection", &m_randomPruningOrderProtection);
        ImGui::DragFloat3("Random pruning base/age/max", &m_randomPruningBaseAgeMax.x, 0.0001f, -1.0f, 1.0f, "%.5f");
        const float maxAgeBeforeMaxCutOff =
                (m_randomPruningBaseAgeMax.z - m_randomPruningBaseAgeMax.x) / m_randomPruningBaseAgeMax.y;
        ImGui::Text("Max age before reaching max: %.3f", maxAgeBeforeMaxCutOff);
        ImGui::DragFloat("Low Branch Pruning", &m_lowBranchPruning, 0.01f);
        ImGui::DragFloat3("Sagging thickness/reduction/max", &m_saggingFactorThicknessReductionMax.x, 0.01f);
        ImGui::TreePop();
    }
    ImGui::DragInt("Mature age", &m_matureAge, 1, 0, 1000);


}

void GeneralTreeParameters::OnCreate() {
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

void GeneralTreeParameters::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_lateralBudCount" << YAML::Value << m_lateralBudCount;
    out << YAML::Key << "m_branchingAngleMeanVariance" << YAML::Value << m_branchingAngleMeanVariance;
    out << YAML::Key << "m_rollAngleMeanVariance" << YAML::Value << m_rollAngleMeanVariance;
    out << YAML::Key << "m_apicalAngleMeanVariance" << YAML::Value << m_apicalAngleMeanVariance;
    out << YAML::Key << "m_gravitropism" << YAML::Value << m_gravitropism;
    out << YAML::Key << "m_phototropism" << YAML::Value << m_phototropism;
    out << YAML::Key << "m_internodeLengthMeanVariance" << YAML::Value << m_internodeLengthMeanVariance;
    out << YAML::Key << "m_endNodeThicknessAndControl" << YAML::Value << m_endNodeThicknessAndControl;
    out << YAML::Key << "m_lateralBudFlushingProbability" << YAML::Value << m_lateralBudFlushingProbability;
    out << YAML::Key << "m_neighborAvoidance" << YAML::Value << m_neighborAvoidance;
    out << YAML::Key << "m_apicalControl" << YAML::Value << m_apicalControl;
    out << YAML::Key << "m_apicalDominanceBaseAgeDist" << YAML::Value << m_apicalDominanceBaseAgeDist;
    out << YAML::Key << "m_lateralBudFlushingLightingFactor" << YAML::Value << m_lateralBudFlushingLightingFactor;
    out << YAML::Key << "m_budKillProbabilityApicalLateral" << YAML::Value << m_budKillProbabilityApicalLateral;
    out << YAML::Key << "m_randomPruningOrderProtection" << YAML::Value << m_randomPruningOrderProtection;
    out << YAML::Key << "m_randomPruningBaseAgeMax" << YAML::Value << m_randomPruningBaseAgeMax;
    out << YAML::Key << "m_lowBranchPruning" << YAML::Value << m_lowBranchPruning;
    out << YAML::Key << "m_saggingFactorThicknessReductionMax" << YAML::Value << m_saggingFactorThicknessReductionMax;
    out << YAML::Key << "m_matureAge" << YAML::Value << m_matureAge;
}

void GeneralTreeParameters::Deserialize(const YAML::Node &in) {
    if(in["m_lateralBudCount"]) m_lateralBudCount = in["m_lateralBudCount"].as<int>();
    if(in["m_branchingAngleMeanVariance"]) m_branchingAngleMeanVariance = in["m_branchingAngleMeanVariance"].as<glm::vec2>();
    if(in["m_rollAngleMeanVariance"]) m_rollAngleMeanVariance = in["m_rollAngleMeanVariance"].as<glm::vec2>();
    if(in["m_apicalAngleMeanVariance"]) m_apicalAngleMeanVariance = in["m_apicalAngleMeanVariance"].as<glm::vec2>();
    if(in["m_gravitropism"]) m_gravitropism = in["m_gravitropism"].as<float>();
    if(in["m_phototropism"]) m_phototropism = in["m_phototropism"].as<float>();
    if(in["m_internodeLengthMeanVariance"]) m_internodeLengthMeanVariance = in["m_internodeLengthMeanVariance"].as<glm::vec2>();
    if(in["m_endNodeThicknessAndControl"]) m_endNodeThicknessAndControl = in["m_endNodeThicknessAndControl"].as<glm::vec2>();
    if(in["m_lateralBudFlushingProbability"]) m_lateralBudFlushingProbability = in["m_lateralBudFlushingProbability"].as<float>();
    if(in["m_neighborAvoidance"]) m_neighborAvoidance = in["m_neighborAvoidance"].as<glm::vec3>();
    if(in["m_apicalControl"]) m_apicalControl = in["m_apicalControl"].as<float>();
    if(in["m_apicalDominanceBaseAgeDist"]) m_apicalDominanceBaseAgeDist = in["m_apicalDominanceBaseAgeDist"].as<glm::vec3>();
    if(in["m_lateralBudFlushingLightingFactor"]) m_lateralBudFlushingLightingFactor = in["m_lateralBudFlushingLightingFactor"].as<float>();
    if(in["m_budKillProbabilityApicalLateral"]) m_budKillProbabilityApicalLateral = in["m_budKillProbabilityApicalLateral"].as<glm::vec2>();
    if(in["m_randomPruningOrderProtection"])  m_randomPruningOrderProtection = in["m_randomPruningOrderProtection"].as<int>();
    if(in["m_randomPruningBaseAgeMax"]) m_randomPruningBaseAgeMax = in["m_randomPruningBaseAgeMax"].as<glm::vec3>();
    if(in["m_lowBranchPruning"]) m_lowBranchPruning = in["m_lowBranchPruning"].as<float>();
    if(in["m_saggingFactorThicknessReductionMax"]) m_saggingFactorThicknessReductionMax = in["m_saggingFactorThicknessReductionMax"].as<glm::vec3>();
    if(in["m_matureAge"]) m_matureAge = in["m_matureAge"].as<int>();
}

Entity GeneralTreeParameters::InstantiateTree() {
    return Application::GetLayer<PlantLayer>()->GetPlantBehaviour<GeneralTreeBehaviour>()->NewPlant(std::dynamic_pointer_cast<GeneralTreeParameters>(m_self.lock()), Transform());
}

void InternodeStatus::OnInspect() {
    ImGui::Text(("Branching Order: " + std::to_string(m_branchingOrder)).c_str());
    ImGui::Text(("Age: " + std::to_string(m_age)).c_str());
    ImGui::Text(("Sagging: " + std::to_string(m_sagging)).c_str());
    ImGui::Text(("Inhibitor: " + std::to_string(m_inhibitor)).c_str());
    ImGui::Text(("DistanceToRoot: " + std::to_string(m_rootDistance)).c_str());
    ImGui::Text(("ChainDistance: " + std::to_string(m_chainDistance)).c_str());
    ImGui::Text(("Dist to start: " + std::to_string(m_branchLength)).c_str());
    ImGui::Text(("MaxDistanceToAnyBranchEnd: " + std::to_string(m_maxDistanceToAnyBranchEnd)).c_str());
    ImGui::Text(("TotalDistanceToAllBranchEnds: " + std::to_string(m_totalDistanceToAllBranchEnds)).c_str());
    ImGui::Text(("Order: " + std::to_string(m_order)).c_str());
    ImGui::Text(("Level: " + std::to_string(m_level)).c_str());
    ImGui::Text(("Biomass: " + std::to_string(m_biomass)).c_str());
    ImGui::Text(("ChildTotalBiomass: " + std::to_string(m_childTotalBiomass)).c_str());
    ImGui::Text(("NodeCount: " + std::to_string(m_currentTotalNodeCount)).c_str());
    ImGui::Text(("Start Density: " + std::to_string(m_startDensity)).c_str());
    glm::vec3 localRotation = glm::eulerAngles(m_desiredLocalRotation);
    ImGui::Text(("Desired local rotation: [" + std::to_string(glm::degrees(localRotation.x)) + ", " + std::to_string(glm::degrees(localRotation.y)) + ", " +std::to_string(glm::degrees(localRotation.z)) + "]").c_str());

}

void InternodeStatus::CalculateApicalControl(float apicalControl) {
    m_apicalControl = glm::pow(1.0f / apicalControl, m_level);
}

void InternodeWaterPressure::OnInspect() {
    ImGui::Text(("m_value: " + std::to_string(m_value)).c_str());
}

void InternodeWater::OnInspect() {
    ImGui::Text(("m_value: " + std::to_string(m_value)).c_str());
}

void InternodeIllumination::OnInspect() {
    ImGui::Text(("Intensity: " + std::to_string(m_intensity)).c_str());
    ImGui::Text(("Direction: [" + std::to_string(glm::degrees(m_direction.x)) + ", " +
                 std::to_string(glm::degrees(m_direction.y)) + ", " + std::to_string(glm::degrees(m_direction.z)) +
                 "]").c_str());
}