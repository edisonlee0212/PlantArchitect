//
// Created by lllll on 8/29/2021.
//

#include "SpaceColonizationBehaviour.hpp"
#include "EmptyInternodeResource.hpp"
#include "CubeVolume.hpp"
#include "InternodeSystem.hpp"

using namespace PlantArchitect;

void SpaceColonizationBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled Space Colonization Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("Space Colonization Internode", InternodeInfo(),
                                                 SpaceColonizationTag(), SpaceColonizationIncentive(), SpaceColonizationParameters(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(SpaceColonizationTag());
}

void SpaceColonizationBehaviour::PreProcess() {
    if (m_attractionPoints.empty()) return;
    std::vector<int> removeMarks;
    removeMarks.resize(m_attractionPoints.size());
    memset(removeMarks.data(), 0, removeMarks.size() * sizeof(bool));
    //1. Check and remove points.
    EntityManager::ForEach<InternodeInfo, GlobalTransform, SpaceColonizationParameters>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &globalTransform,
                 SpaceColonizationParameters &spaceColonizationParameters) {
                 glm::vec3 position = globalTransform.GetPosition() +
                                      internodeInfo.m_length * (globalTransform.GetRotation() * glm::vec3(0, 0, -1));
                 int index = 0;
                 for (const auto &point: m_attractionPoints) {
                     const glm::vec3 diff = position - point;
                     const float distance2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                     if (distance2 <
                         spaceColonizationParameters.m_removeDistance * spaceColonizationParameters.m_removeDistance) {
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
}

void SpaceColonizationBehaviour::Grow() {
    if (m_attractionPoints.empty()) return;
    //1. Allocate near points
    std::vector<std::pair<Entity, float>> minDistance;
    minDistance.resize(m_attractionPoints.size());
    std::mutex distanceMutex;
    for (auto &i: minDistance) {
        i.first = Entity();
        i.second = 999999;
    }
    EntityManager::ForEach<InternodeInfo, GlobalTransform, SpaceColonizationIncentive, SpaceColonizationParameters>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &globalTransform,
                 SpaceColonizationIncentive &spaceColonizationIncentive,
                 SpaceColonizationParameters &spaceColonizationParameters) {
                 glm::vec3 position = globalTransform.GetPosition() +
                                      internodeInfo.m_length * (globalTransform.GetRotation() * glm::vec3(0, 0, -1));
                 spaceColonizationIncentive.m_direction = glm::vec3(0.0f);
                 spaceColonizationIncentive.m_pointAmount = 0;
                 glm::vec3 front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
                 for (int index = 0; index < m_attractionPoints.size(); index++) {
                     auto &point = m_attractionPoints[index];
                     const glm::vec3 diff = point - position;
                     if(glm::dot(diff, front) <= 0) continue;
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
        auto parameter = entity.GetDataComponent<SpaceColonizationParameters>();
        auto position = globalTransform.GetPosition();
        auto front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
        glm::vec3 newPosition = position + parameter.m_internodeLength * front;
        auto incentive = entity.GetDataComponent<SpaceColonizationIncentive>();
        incentive.m_direction += m_attractionPoints[i] - newPosition;
        incentive.m_pointAmount++;
        entity.SetDataComponent(incentive);
    }
    //2. Form new internodes.
    std::vector<Entity> entities;
    m_internodesQuery.ToEntityArray(entities);
    for (const auto &entity: entities) {
        if (!entity.IsEnabled()) continue;
        auto parameter = entity.GetDataComponent<SpaceColonizationParameters>();
        auto globalTransform = entity.GetDataComponent<GlobalTransform>();
        auto tag = entity.GetDataComponent<SpaceColonizationTag>();
        auto position = globalTransform.GetPosition();
        auto front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
        auto up = globalTransform.GetRotation() * glm::vec3(0, 1, 0);
        auto spaceColonizationIncentive = entity.GetDataComponent<SpaceColonizationIncentive>();
        Entity newNode;
        glm::vec3 newPosition = position + parameter.m_internodeLength * front;
        glm::vec3 newFront;
        if (spaceColonizationIncentive.m_pointAmount != 0) {
            if(glm::all(glm::equal(spaceColonizationIncentive.m_direction, glm::vec3(0)))){
                continue;
            }
            newFront = glm::normalize(spaceColonizationIncentive.m_direction);
            bool duplicate = false;
            entity.ForEachChild([&](Entity child){
                if(glm::dot(child.GetDataComponent<GlobalTransform>().GetRotation() * glm::vec3(0, 0, -1), newFront) > 0.95f) duplicate = true;
            });
            if(duplicate) continue;
            newNode = Retrieve<EmptyInternodeResource>(entity);
            tag.m_truck = false;
            newNode.SetDataComponent(tag);
            entity.SetDataComponent(tag);
        } else if (tag.m_truck) {
            newNode = Retrieve<EmptyInternodeResource>(entity);
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
        newNode.SetDataComponent(parameter);
        InternodeInfo newInfo;
        newInfo.m_length = parameter.m_internodeLength;
        newInfo.m_thickness = parameter.m_endNodeThickness;
        newNode.SetDataComponent(newInfo);
        auto newInternode = newNode.GetOrSetPrivateComponent<Internode>().lock();
    }
}

void SpaceColonizationBehaviour::PostProcess() {
    std::vector<Entity> plants;
    CollectRoots(m_internodesQuery, plants);
    int plantSize = plants.size();

    //Use internal JobSystem to dispatch job for entity collection.
    std::vector<std::shared_future<void>> results;
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        results.push_back(JobManager::PrimaryWorkers().Push([&, plantIndex](int id) {
            TreeGraphWalkerEndToRoot(plants[plantIndex], plants[plantIndex], [&](Entity parent){
                float thicknessCollection = 0.0f;
                auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                auto parameters = parent.GetDataComponent<SpaceColonizationParameters>();
                parent.ForEachChild([&](Entity child){
                    if(!InternodeCheck(child)) return;
                    auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                    thicknessCollection += glm::pow(childInternodeInfo.m_thickness, 1.0f / parameters.m_thicknessFactor);
                });
                parentInternodeInfo.m_thickness = glm::pow(thicknessCollection, parameters.m_thicknessFactor);
                parent.SetDataComponent(parentInternodeInfo);
            }, [](Entity endNode){
                auto internodeInfo = endNode.GetDataComponent<InternodeInfo>();
                auto parameters = endNode.GetDataComponent<SpaceColonizationParameters>();
                internodeInfo.m_thickness = parameters.m_endNodeThickness;
                endNode.SetDataComponent(internodeInfo);
            });
        }).share());
    }
    for (const auto &i: results)
        i.wait();
}

void SpaceColonizationBehaviour::OnInspect() {
    RecycleButton();

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
                 auto entity = Retrieve<EmptyInternodeResource>();
                 Transform internodeTransform;
                 internodeTransform.m_value =
                         glm::translate(glm::vec3(0.0f)) *
                         glm::mat4_cast(glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f))) *
                         glm::scale(glm::vec3(1.0f));
                 internodeTransform.m_value = transform.m_value * internodeTransform.m_value;
                 entity.SetDataComponent(internodeTransform);
                 SpaceColonizationTag tag;
                 tag.m_truck = true;
                 entity.SetDataComponent(tag);
                 InternodeInfo newInfo;
                 newInfo.m_length = params.m_internodeLength;
                 newInfo.m_thickness = params.m_endNodeThickness;
                 entity.SetDataComponent(newInfo);
                 entity.SetDataComponent(params);
                 return entity;
             }
            );

    static float resolution = 0.02;
    static float subdivision = 4.0;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate branch mesh")) {
        GenerateBranchSkinnedMeshes(m_internodesQuery, subdivision, resolution);
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
            if (EditorManager::DragAndDropButton<Volume>(volume, "Slot " + std::to_string(index++))) {
                skip = true;
                break;
            }
            ImGui::TreePush();
            if (ImGui::Button("Add attraction points")) {
                for (int i = 0; i < amount; i++) {
                    m_attractionPoints.push_back(volume.Get<Volume>()->GetRandomPoint());
                }
            }
            ImGui::TreePop();
        }
        if (skip) {
            int index = 0;
            for (auto &i: m_volumes) {
                if (!i.Get<Volume>()) {
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
            RenderManager::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube, renderColor,
                                                  displayMatrices, glm::mat4(1.0f), renderSize);
            RenderManager::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube,
                                                  EntityManager::GetSystem<InternodeSystem>()->m_internodeDebuggingCamera,
                                                  EditorManager::GetInstance().m_sceneCameraPosition,
                                                  EditorManager::GetInstance().m_sceneCameraRotation, renderColor,
                                                  displayMatrices, glm::mat4(1.0f), renderSize);
        }
        ImGui::TreePop();
    }
}

void SpaceColonizationBehaviour::VolumeSlotButton() {
    ImGui::Text("Drop Volume");
    ImGui::SameLine();
    ImGui::Button("Here");
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("CubeVolume")) {
            IM_ASSERT(payload->DataSize == sizeof(std::shared_ptr<IPrivateComponent>));
            std::shared_ptr<Volume> payload_n =
                    std::dynamic_pointer_cast<Volume>(
                            *static_cast<std::shared_ptr<IPrivateComponent> *>(payload->Data));
            PushVolume(payload_n);
        }
        ImGui::EndDragDropTarget();
    }
}

void SpaceColonizationBehaviour::PushVolume(const std::shared_ptr<Volume>& volume) {
    if(!volume.get()) return;
    bool search = false;
    for (auto &i: m_volumes) {
        if (i.Get<Volume>()->GetTypeName() == volume->GetTypeName()) search = true;
    }
    if (!search) {
        m_volumes.emplace_back(volume);
    }
}

bool SpaceColonizationBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<SpaceColonizationTag>();
}


void SpaceColonizationParameters::OnInspect() {
    ImGui::DragFloat("Remove Distance", &m_removeDistance);
    ImGui::DragFloat("Attract Distance", &m_attractDistance);
    ImGui::DragFloat("Internode Length", &m_internodeLength);
    ImGui::DragFloat("Thickness Factor", &m_thicknessFactor);
    ImGui::DragFloat("End node thickness", &m_endNodeThickness);
}
