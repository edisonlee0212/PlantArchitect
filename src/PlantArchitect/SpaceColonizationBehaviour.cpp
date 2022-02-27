//
// Created by lllll on 8/29/2021.
//

#include "SpaceColonizationBehaviour.hpp"
#include "EmptyInternodeResource.hpp"
#include "CubeVolume.hpp"
#include "InternodeLayer.hpp"
#include "TransformLayer.hpp"
#include "EditorLayer.hpp"

using namespace PlantArchitect;

void SpaceColonizationBehaviour::OnCreate() {
    m_internodeArchetype =
            Entities::CreateEntityArchetype("Space Colonization Internode", InternodeInfo(), InternodeStatistics(),
                                            SpaceColonizationTag(), SpaceColonizationIncentive(),
                                            BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                            BranchPointer());
    m_internodesQuery = Entities::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo(), SpaceColonizationTag());

    m_rootArchetype =
            Entities::CreateEntityArchetype("Space Colonization Root", RootInfo(),
                                            SpaceColonizationTag(),
                                            SpaceColonizationParameters());
    m_rootsQuery = Entities::CreateEntityQuery();
    m_rootsQuery.SetAllFilters(RootInfo(), SpaceColonizationTag());

    m_branchArchetype =
            Entities::CreateEntityArchetype("Space Colonization Branch", BranchInfo(),
                                            SpaceColonizationTag());
    m_branchesQuery = Entities::CreateEntityQuery();
    m_branchesQuery.SetAllFilters(BranchInfo(), SpaceColonizationTag());
    m_volumes.clear();
}

void SpaceColonizationBehaviour::Grow(int iteration) {

    if (m_attractionPoints.empty()) return;
    std::vector<int> removeMarks;
    removeMarks.resize(m_attractionPoints.size());
    memset(removeMarks.data(), 0, removeMarks.size() * sizeof(bool));
    //1. Check and remove points.
    Entities::ForEach<InternodeInfo, GlobalTransform>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &globalTransform) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto rootEntity = internode->m_currentRoot.Get();
                 if (!RootCheck(rootEntity)) return;
                 auto spaceColonizationParameters = rootEntity.GetDataComponent<SpaceColonizationParameters>();
                 glm::vec3 position = globalTransform.GetPosition() +
                                      internodeInfo.m_length *
                                      (globalTransform.GetRotation() * glm::vec3(0, 0, -1));
                 int index = 0;
                 for (const auto &point: m_attractionPoints) {
                     const glm::vec3 diff = position - point;
                     const float distance2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                     if (distance2 <
                         spaceColonizationParameters.m_removeDistance *
                         spaceColonizationParameters.m_removeDistance) {
                         removeMarks[index] = 1;
                     }
                     index++;
                 }
             }, true);
    std::vector<glm::vec3> newAttractionPoints;
    for (int i = 0; i < m_attractionPoints.size(); i++) {
        if (removeMarks[i] == 0) {
            newAttractionPoints.push_back(m_attractionPoints[i]);
        }
    }
    std::swap(newAttractionPoints, m_attractionPoints);
    //2. Calculate center
    m_center = glm::vec3(0.0f);
    for (const auto &i: m_attractionPoints) {
        m_center += i;
    }
    m_center /= m_attractionPoints.size();

    //1. Allocate near points
    std::vector<std::pair<Entity, float>> minDistance;
    minDistance.resize(m_attractionPoints.size());
    std::mutex distanceMutex;
    for (auto &i: minDistance) {
        i.first = Entity();
        i.second = 999999;
    }
    Entities::ForEach<InternodeInfo, GlobalTransform, SpaceColonizationIncentive>
            (Entities::GetCurrentScene(), Jobs::Workers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &globalTransform,
                 SpaceColonizationIncentive &spaceColonizationIncentive) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto rootEntity = internode->m_currentRoot.Get();
                 if (!RootCheck(rootEntity)) return;
                 auto spaceColonizationParameters = rootEntity.GetDataComponent<SpaceColonizationParameters>();
                 glm::vec3 position = globalTransform.GetPosition() +
                                      internodeInfo.m_length *
                                      (globalTransform.GetRotation() * glm::vec3(0, 0, -1));
                 spaceColonizationIncentive.m_direction = glm::vec3(0.0f);
                 spaceColonizationIncentive.m_pointAmount = 0;
                 glm::vec3 front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
                 for (int index = 0; index < m_attractionPoints.size(); index++) {
                     auto &point = m_attractionPoints[index];
                     const glm::vec3 diff = point - position;
                     if (glm::dot(diff, front) <= 0) continue;
                     const float distance2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                     if (distance2 < spaceColonizationParameters.m_attractDistance *
                                     spaceColonizationParameters.m_attractDistance) {
                         std::lock_guard<std::mutex> lock(distanceMutex);
                         if (distance2 < minDistance[index].second) {
                             minDistance[index].first = entity;
                             minDistance[index].second = distance2;
                         }
                     }
                 }
             }, true);
    for (int i = 0; i < minDistance.size(); i++) {
        auto entity = minDistance[i].first;
        if (entity.IsNull()) continue;
        auto globalTransform = entity.GetDataComponent<GlobalTransform>();
        auto internodeInfo = entity.GetDataComponent<InternodeInfo>();
        auto position = globalTransform.GetPosition();
        auto front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
        glm::vec3 newPosition = position + internodeInfo.m_length * front;
        auto incentive = entity.GetDataComponent<SpaceColonizationIncentive>();
        incentive.m_direction += m_attractionPoints[i] - newPosition;
        incentive.m_pointAmount++;
        entity.SetDataComponent(incentive);
    }
    //2. Form new internodes.
    std::vector<Entity> entities;
    m_internodesQuery.ToEntityArray(Entities::GetCurrentScene(), entities);
    for (const auto &entity: entities) {
        auto internodeInfo = entity.GetDataComponent<InternodeInfo>();
        auto globalTransform = entity.GetDataComponent<GlobalTransform>();
        auto tag = entity.GetDataComponent<SpaceColonizationTag>();
        auto position = globalTransform.GetPosition();
        auto front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
        auto up = globalTransform.GetRotation() * glm::vec3(0, 1, 0);
        auto spaceColonizationIncentive = entity.GetDataComponent<SpaceColonizationIncentive>();
        Entity newNode;
        glm::vec3 newPosition = position + internodeInfo.m_length * front;
        glm::vec3 newFront;
        if (spaceColonizationIncentive.m_pointAmount != 0) {
            if (glm::all(glm::equal(spaceColonizationIncentive.m_direction, glm::vec3(0)))) {
                continue;
            }
            newFront = glm::normalize(spaceColonizationIncentive.m_direction);
            bool duplicate = false;
            entity.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
                if (glm::dot(child.GetDataComponent<GlobalTransform>().GetRotation() * glm::vec3(0, 0, -1),
                             newFront) > 0.95f)
                    duplicate = true;
            });
            if (duplicate) continue;
            newNode = CreateInternode(entity);
            tag.m_truck = false;
            newNode.SetDataComponent(tag);
            entity.SetDataComponent(tag);
        } else if (tag.m_truck) {
            newNode = CreateInternode(entity);
            newFront = glm::normalize(m_center - newPosition);
            newNode.SetDataComponent(tag);
            tag.m_truck = false;
            entity.SetDataComponent(tag);
        } else continue;
        glm::vec3 newUp = glm::cross(glm::cross(newFront, up), newFront);
        GlobalTransform newNodeGlobalTransform;
        newNodeGlobalTransform.m_value =
                glm::translate(newPosition) * glm::mat4_cast(glm::quatLookAt(newFront, newUp)) *
                glm::scale(glm::vec3(1.0f));
        newNode.SetDataComponent(newNodeGlobalTransform);
        auto newInternode = newNode.GetOrSetPrivateComponent<Internode>().lock();
        auto parameter = newInternode->m_currentRoot.Get().GetDataComponent<SpaceColonizationParameters>();
        InternodeInfo newInfo;
        newInfo.m_length = glm::gaussRand(parameter.m_internodeLengthMean, parameter.m_internodeLengthVariance);
        newInfo.m_thickness = parameter.m_endNodeThickness;
        newNode.SetDataComponent(newInfo);
    }
    std::vector<Entity> currentRoots;
    m_rootsQuery.ToEntityArray(Entities::GetCurrentScene(), currentRoots);
    int plantSize = currentRoots.size();

    //Use internal JobSystem to dispatch job for entity collection.
    std::vector<std::shared_future<void>> results;
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        results.push_back(Jobs::Workers().Push([&, plantIndex](int id) {
            auto root = currentRoots[plantIndex];
            auto parent = root.GetParent();
            if (!parent.IsNull()) {
                auto globalTransform = root.GetDataComponent<GlobalTransform>();
                auto parentGlobalTransform = parent.GetDataComponent<GlobalTransform>();
                Transform rootLocalTransform;
                rootLocalTransform.m_value = glm::inverse(parentGlobalTransform.m_value) * globalTransform.m_value;
                root.SetDataComponent(rootLocalTransform);
            }
            Application::GetLayer<TransformLayer>()->CalculateTransformGraphForDescendents(
                    Entities::GetCurrentScene(), root);
        }).share());
    }
    for (const auto &i: results)
        i.wait();

    Entities::ForEach<GlobalTransform>(Entities::GetCurrentScene(), Jobs::Workers(), m_rootsQuery,
                                       [&](int i, Entity entity, GlobalTransform &globalTransform) {
                                           if (!RootCheck(entity)) return;
                                           auto spaceColonizationParameters = entity.GetDataComponent<SpaceColonizationParameters>();
                                           entity.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
                                               if (!InternodeCheck(child)) return;
                                               InternodeGraphWalkerEndToRoot(child, [&](Entity parent) {
                                                   float thicknessCollection = 0.0f;
                                                   auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                                                   parent.ForEachChild(
                                                           [&](const std::shared_ptr<Scene> &scene, Entity child) {
                                                               if (!InternodeCheck(child)) return;
                                                               auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                                                               thicknessCollection += glm::pow(
                                                                       childInternodeInfo.m_thickness,
                                                                       1.0f /
                                                                       spaceColonizationParameters.m_thicknessFactor);
                                                           });
                                                   parentInternodeInfo.m_thickness = glm::pow(thicknessCollection,
                                                                                              spaceColonizationParameters.m_thicknessFactor);
                                                   parent.SetDataComponent(parentInternodeInfo);
                                               }, [&](Entity endNode) {
                                                   auto internodeInfo = endNode.GetDataComponent<InternodeInfo>();
                                                   internodeInfo.m_thickness = spaceColonizationParameters.m_endNodeThickness;
                                                   endNode.SetDataComponent(internodeInfo);
                                               });
                                           });

                                       });

}

void SpaceColonizationBehaviour::OnInspect() {
    CreateInternodeMenu<SpaceColonizationParameters>
            ("New Space Colonization Plant Wizard",
             ".scparams",
             [](SpaceColonizationParameters &params) {
                 params.OnInspect();
             },
             [](SpaceColonizationParameters &params,
                const std::filesystem::path &path) {

             },
             [](const SpaceColonizationParameters &params,
                const std::filesystem::path &path) {

             },
             [&](const SpaceColonizationParameters &params,
                 const Transform &transform) {
                 return NewPlant(params, transform);
             }
            );

    static float resolution = 0.02;
    static float subdivision = 4.0;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate meshes")) {
        GenerateSkinnedMeshes(subdivision, resolution);
    }

    if (ImGui::TreeNodeEx("Attraction Points", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Clear")) {
            m_attractionPoints.clear();
        }
        VolumeSlotButton();
        int index = 0;
        bool skip = false;
        static int amount = 1000;
        ImGui::DragInt("Amount", &amount);
        for (auto &volume: m_volumes) {
            if (Editor::DragAndDropButton<IVolume>(volume, "Slot " + std::to_string(index++))) {
                skip = true;
                break;
            }
            ImGui::TreePush();
            if (ImGui::Button("Add attraction points")) {
                for (int i = 0; i < amount; i++) {
                    m_attractionPoints.push_back(volume.Get<IVolume>()->GetRandomPoint());
                }
            }
            ImGui::TreePop();
        }
        if (skip) {
            int index = 0;
            for (auto &i: m_volumes) {
                if (!i.Get<IVolume>()) {
                    m_volumes.erase(m_volumes.begin() + index);
                    break;
                }
                index++;
            }
        }
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Growth", ImGuiTreeNodeFlags_DefaultOpen)) {
        static bool renderAttractionPoints = true;
        ImGui::Checkbox("Display attraction points", &renderAttractionPoints);
        if (renderAttractionPoints) {
            static std::vector<glm::mat4> displayMatrices;
            static float renderSize = 0.05f;
            static glm::vec4 renderColor = glm::vec4(0.8f);
            ImGui::DragFloat("Point Size", &renderSize, 0.01f);
            ImGui::ColorEdit4("Point Color", &renderColor.x);
            if (m_attractionPoints.size() != displayMatrices.size()) {
                displayMatrices.resize(m_attractionPoints.size());
                for (int i = 0; i < m_attractionPoints.size(); i++) {
                    displayMatrices[i] = glm::translate(m_attractionPoints[i]) * glm::scale(glm::vec3(1.0f));
                }
            }

            Graphics::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube, renderColor,
                                             displayMatrices, glm::mat4(1.0f), renderSize);
            auto editorLayer = Application::GetLayer<EditorLayer>();
            auto internodeLayer = Application::GetLayer<InternodeLayer>();
            if (editorLayer && internodeLayer) {
                Graphics::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube,
                                                 internodeLayer->m_internodeDebuggingCamera,
                                                 editorLayer->m_sceneCameraPosition,
                                                 editorLayer->m_sceneCameraRotation, renderColor,
                                                 displayMatrices, glm::mat4(1.0f), renderSize);
            }
        }
        ImGui::TreePop();
    }
}

void SpaceColonizationBehaviour::VolumeSlotButton() {
    ImGui::Text("Add new volume");
    ImGui::SameLine();
    static PrivateComponentRef temp;
    Editor::DragAndDropButton(temp, "Here", {"CubeVolume", "RadialBoundingVolume"}, false);
    if (temp.Get<IVolume>()) {
        PushVolume(temp.Get<IVolume>());
        temp.Clear();
    }
}

void SpaceColonizationBehaviour::PushVolume(const std::shared_ptr<IVolume> &volume) {
    if (!volume.get()) return;
    bool search = false;
    for (auto &i: m_volumes) {
        if (i.Get<IVolume>()->GetTypeName() == volume->GetTypeName()) search = true;
    }
    if (!search) {
        m_volumes.emplace_back(volume);
    }
}

bool SpaceColonizationBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<SpaceColonizationTag>();
}

Entity SpaceColonizationBehaviour::CreateInternode(const Entity &parent) {
    return CreateInternodeHelper<EmptyInternodeResource>(parent);
}

Entity SpaceColonizationBehaviour::NewPlant(const SpaceColonizationParameters &params, const Transform &transform) {
    Entity rootInternode, rootBranch;
    auto root = CreateRoot(rootInternode, rootBranch);
    Transform internodeTransform;
    internodeTransform.m_value =
            glm::translate(glm::vec3(0.0f)) *
            glm::mat4_cast(glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f))) *
            glm::scale(glm::vec3(1.0f));
    internodeTransform.m_value = transform.m_value * internodeTransform.m_value;
    rootInternode.SetDataComponent(internodeTransform);
    SpaceColonizationTag tag;
    tag.m_truck = true;
    rootInternode.SetDataComponent(tag);
    InternodeInfo newInfo;
    newInfo.m_length = glm::gaussRand(params.m_internodeLengthMean, params.m_internodeLengthVariance);
    newInfo.m_thickness = params.m_endNodeThickness;
    rootInternode.SetDataComponent(newInfo);
    root.SetDataComponent(params);
    return root;
}

void SpaceColonizationBehaviour::Serialize(YAML::Emitter &out) {

}

void SpaceColonizationBehaviour::Deserialize(const YAML::Node &in) {
}

bool SpaceColonizationBehaviour::InternalRootCheck(const Entity &target) {
    return target.HasDataComponent<SpaceColonizationTag>();
}

bool SpaceColonizationBehaviour::InternalBranchCheck(const Entity &target) {
    return target.HasDataComponent<SpaceColonizationTag>();
}

Entity SpaceColonizationBehaviour::CreateRoot(Entity &rootInternode, Entity &rootBranch) {
    return CreateRootHelper<EmptyInternodeResource>(rootInternode, rootBranch);
}

Entity SpaceColonizationBehaviour::CreateBranch(const Entity &parent) {
    return CreateBranchHelper(parent);
}

void SpaceColonizationParameters::OnInspect() {
    ImGui::DragFloat("Remove Distance", &m_removeDistance);
    ImGui::DragFloat("Attract Distance", &m_attractDistance);
    ImGui::DragFloat2("Internode Length Mean/Var", &m_internodeLengthMean);
    ImGui::DragFloat("Thickness Factor", &m_thicknessFactor);
    ImGui::DragFloat("End node thickness", &m_endNodeThickness);
}

void SpaceColonizationParameters::Save(const std::filesystem::path &path) const {

}

void SpaceColonizationParameters::Load(const std::filesystem::path &path) {

}