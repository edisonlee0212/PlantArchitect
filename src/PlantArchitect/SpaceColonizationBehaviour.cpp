//
// Created by lllll on 8/29/2021.
//

#include "SpaceColonizationBehaviour.hpp"
#include "DefaultInternodeResource.hpp"
#include "CubeVolume.hpp"
#include "InternodeSystem.hpp"
using namespace PlantArchitect;

void SpaceColonizationBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled Space Colonization Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("Space Colonization Internode", InternodeInfo(),
                                                 SpaceColonizationTag(), SpaceColonizationIncentive(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo());
}

void SpaceColonizationBehaviour::PreProcess() {
    if (m_attractionPoints.empty()) return;
    std::vector<int> removeMarks;
    removeMarks.resize(m_attractionPoints.size());
    memset(removeMarks.data(), 0, removeMarks.size() * sizeof(bool));
    //1. Check and remove points.
    EntityManager::ForEach<InternodeInfo, GlobalTransform>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &globalTransform) {
                 glm::vec3 position = globalTransform.GetPosition() +
                                      internodeInfo.m_length * (globalTransform.GetRotation() * glm::vec3(0, 0, -1));
                 int index = 0;
                 for (const auto &point: m_attractionPoints) {
                     const glm::vec3 diff = position - point;
                     const float distance2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                     if (distance2 < m_removeDistance * m_removeDistance) {
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
    EntityManager::ForEach<InternodeInfo, GlobalTransform, SpaceColonizationIncentive>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeInfo, GlobalTransform &globalTransform,
                 SpaceColonizationIncentive &spaceColonizationIncentive) {
                 glm::vec3 position = globalTransform.GetPosition() +
                                      internodeInfo.m_length * (globalTransform.GetRotation() * glm::vec3(0, 0, -1));
                 spaceColonizationIncentive.m_direction = glm::vec3(0.0f);
                 spaceColonizationIncentive.m_pointAmount = 0;
                 for (int index = 0; index < m_attractionPoints.size(); index++) {
                     auto &point = m_attractionPoints[index];
                     const glm::vec3 diff = position - point;
                     const float distance2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                     if (distance2 < m_attractDistance * m_attractDistance) {
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
        auto position = globalTransform.GetPosition();
        auto front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
        glm::vec3 newPosition = position + m_internodeLength * front;
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
        auto globalTransform = entity.GetDataComponent<GlobalTransform>();
        auto tag = entity.GetDataComponent<SpaceColonizationTag>();
        auto position = globalTransform.GetPosition();
        auto front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
        auto up = globalTransform.GetRotation() * glm::vec3(0, 1, 0);
        auto spaceColonizationIncentive = entity.GetDataComponent<SpaceColonizationIncentive>();
        Entity newNode;
        glm::vec3 newPosition = position + m_internodeLength * front;
        glm::vec3 newFront;
        if (spaceColonizationIncentive.m_pointAmount != 0) {
            newNode = Retrieve<DefaultInternodeResource>(entity);
            newFront = glm::normalize(spaceColonizationIncentive.m_direction);
            tag.m_truck = false;
            newNode.SetDataComponent(tag);
        }else if(tag.m_truck){
            newNode = Retrieve<DefaultInternodeResource>(entity);
            newFront = glm::normalize(m_center - newPosition);
            newNode.SetDataComponent(tag);
        }else return;
        glm::vec3 newUp = glm::cross(glm::cross(newFront, up), newFront);
        GlobalTransform newNodeGlobalTransform;
        newNodeGlobalTransform.m_value =
                glm::translate(newPosition) * glm::mat4_cast(glm::quatLookAt(newFront, newUp)) *
                glm::scale(glm::vec3(1.0f));
        newNode.SetDataComponent(newNodeGlobalTransform);

        InternodeInfo newInfo;
        newInfo.m_length = m_internodeLength;
        newInfo.m_thickness = m_internodeLength / 10.0f;
        newNode.SetDataComponent(newInfo);
        auto newInternode = newNode.GetOrSetPrivateComponent<Internode>().lock();
        if(entity.GetChildrenAmount() > 0){
            newInternode->m_fromApicalBud = false;
        }else{
            newInternode->m_fromApicalBud = true;
        }
    }
}

void SpaceColonizationBehaviour::PostProcess() {

}

void SpaceColonizationBehaviour::OnInspect() {
    if (ImGui::Button("Create new plant...")) {
        auto entity = Retrieve<DefaultInternodeResource>();
        Transform internodeTransform;
        internodeTransform.m_value =
                glm::translate(glm::vec3(0.0f)) *
                glm::mat4_cast(glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f))) *
                glm::scale(glm::vec3(1.0f));
        entity.SetDataComponent(internodeTransform);
        SpaceColonizationTag tag;
        tag.m_truck = true;
        entity.SetDataComponent(tag);
    }
    if (ImGui::Button("Generate branch mesh")) {
        std::vector<Entity> entities;
        m_internodesQuery.ToEntityArray(entities);
        GenerateBranchSkinnedMeshes(entities);
    }

    if (ImGui::TreeNodeEx("Volumes", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            RenderManager::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube, EntityManager::GetSystem<InternodeSystem>()->m_internodeDebuggingCamera,
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
            std::shared_ptr<CubeVolume> payload_n =
                    std::dynamic_pointer_cast<CubeVolume>(
                            *static_cast<std::shared_ptr<IPrivateComponent> *>(payload->Data));
            if (payload_n.get()) {
                bool search = false;
                for (auto &i: m_volumes) {
                    if (i.Get<Volume>()->GetTypeName() == "CubeVolume") search = true;
                }
                if (!search) {
                    m_volumes.emplace_back(payload_n);
                }
            }
        }
        ImGui::EndDragDropTarget();
    }
}


