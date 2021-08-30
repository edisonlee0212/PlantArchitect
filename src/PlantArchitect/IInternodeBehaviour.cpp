//
// Created by lllll on 8/27/2021.
//

#include "IInternodeBehaviour.hpp"
#include "Internode.hpp"

using namespace PlantArchitect;

void IInternodeBehaviour::Recycle(const Entity &internode) {
    auto children = internode.GetChildren();
    if (children.empty()) RecycleSingle(internode);
    else {
        for (const auto &child: children) {
            Recycle(child);
        }
    }
}

void IInternodeBehaviour::RecycleSingle(const Entity &internode) {
    std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
    internode.GetOrSetPrivateComponent<Internode>().lock()->OnRecycle();
    internode.SetParent(m_recycleStorageEntity.Get());
    internode.SetEnabled(false);
    m_recycledInternodes.emplace_back(internode);
}

void IInternodeBehaviour::GenerateBranchSkinnedMeshes(const EntityQuery &internodeQuery) {
    std::mutex plantCollectionMutex;
    std::vector<Entity> plants;
    EntityManager::ForEach<InternodeInfo>(JobManager::PrimaryWorkers(), internodeQuery,
                                          [&](int index, Entity entity, InternodeInfo &internodeInfo) {
                                              if (!entity.HasPrivateComponent<Internode>()) return;
                                              auto parent = entity.GetParent();
                                              if (parent.IsNull() || !parent.HasPrivateComponent<Internode>()) {
                                                  std::lock_guard<std::mutex> lock(plantCollectionMutex);
                                                  plants.push_back(entity);
                                              }
                                          });

    int plantSize = plants.size();
    std::vector<std::vector<Entity>> boundEntitiesLists;
    std::vector<std::vector<unsigned>> boneIndicesLists;
    std::vector<std::vector<int>> parentIndicesLists;
    boundEntitiesLists.resize(plantSize);
    boneIndicesLists.resize(plantSize);
    parentIndicesLists.resize(plantSize);

    //Use internal JobSystem to dispatch job for entity collection.
    std::vector<std::shared_future<void>> results;
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        results.push_back(JobManager::PrimaryWorkers().Push([&](int id) {
            TreeNodeWalker(boundEntitiesLists[plantIndex],
                           parentIndicesLists[plantIndex], -1, plants[plantIndex]);
        }).share());
    }
    for (const auto &i: results)
        i.wait();

    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        const auto &plant = plants[plantIndex];
        auto animator = plant.GetOrSetPrivateComponent<Animator>().lock();
        auto skinnedMeshRenderer = plant.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
        skinnedMeshRenderer->SetEnabled(true);
        if (plant.HasPrivateComponent<MeshRenderer>()) {
            plant.GetOrSetPrivateComponent<MeshRenderer>().lock()->SetEnabled(false);
        }
        auto treeData = plant.GetOrSetPrivateComponent<Internode>().lock();
        const auto plantGlobalTransform =
                plant.GetDataComponent<GlobalTransform>();
#pragma region Branch mesh
#pragma region Animator
        std::vector<glm::mat4> offsetMatrices;
        std::vector<std::string> names;
        offsetMatrices.resize(boundEntitiesLists[plantIndex].size());
        names.resize(boundEntitiesLists[plantIndex].size());
        boneIndicesLists[plantIndex].resize(
                boundEntitiesLists[plantIndex].size());
        for (int i = 0; i < boundEntitiesLists[plantIndex].size(); i++) {
            names[i] = boundEntitiesLists[plantIndex][i].GetName();
            offsetMatrices[i] =
                    glm::inverse(glm::inverse(plantGlobalTransform.m_value) *
                                 boundEntitiesLists[plantIndex][i]
                                         .GetDataComponent<GlobalTransform>()
                                         .m_value);
            boneIndicesLists[plantIndex][i] = i;
        }
        animator->Setup(boundEntitiesLists[plantIndex], names, offsetMatrices);
#pragma endregion
        std::vector<unsigned> skinnedIndices;
        std::vector<SkinnedVertex> skinnedVertices;
        TreeSkinnedMeshGenerator(boundEntitiesLists[plantIndex],
                                 parentIndicesLists[plantIndex],
                                 skinnedVertices, skinnedIndices);
        treeData->m_skinnedBranchMesh.Get<SkinnedMesh>()->SetVertices(
                17, skinnedVertices, skinnedIndices);
        treeData->m_skinnedBranchMesh.Get<SkinnedMesh>()
                ->m_boneAnimatorIndices = boneIndicesLists[plantIndex];
        skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(
                treeData->m_skinnedBranchMesh.Get<SkinnedMesh>());

#pragma endregion

    }
}

void IInternodeBehaviour::TreeNodeWalker(std::vector<Entity> &boundEntities, std::vector<int> &parentIndices,
                                         const int &parentIndex, const Entity &node) {
    if (!node.HasDataComponent<InternodeInfo>() || !node.HasPrivateComponent<Internode>()) return;
    boundEntities.push_back(node);
    parentIndices.push_back(parentIndex);
    const size_t currentIndex = boundEntities.size() - 1;
    auto info = node.GetDataComponent<InternodeInfo>();
    info.m_index = currentIndex;
    node.SetDataComponent(info);
    node.ForEachChild([&](Entity child) {
        TreeNodeWalker(boundEntities, parentIndices, currentIndex, child);
    });
}

void IInternodeBehaviour::TreeSkinnedMeshGenerator(std::vector<Entity> &internodes, std::vector<int> &parentIndices,
                                                   std::vector<SkinnedVertex> &vertices,
                                                   std::vector<unsigned int> &indices) {
    int parentStep = -1;
    for (int internodeIndex = 1; internodeIndex < internodes.size();
         internodeIndex++) {
        auto &internode = internodes[internodeIndex];
        glm::vec3 newNormalDir = internodes[parentIndices[internodeIndex]]
                .GetOrSetPrivateComponent<Internode>()
                .lock()
                ->m_normalDir;
        const glm::vec3 front =
                internode.GetDataComponent<GlobalTransform>().GetRotation() *
                glm::vec3(0.0f, 0.0f, -1.0f);
        newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);
        auto list = internode.GetOrSetPrivateComponent<Internode>().lock();
        if (list->m_rings.empty()) {
            continue;
        }
        auto step = list->m_step;
        // For stitching
        const int pStep = parentStep > 0 ? parentStep : step;
        parentStep = step;
        list->m_normalDir = newNormalDir;
        float angleStep = 360.0f / static_cast<float>(pStep);
        int vertexIndex = vertices.size();
        SkinnedVertex archetype;
        float textureXStep = 1.0f / pStep * 4.0f;

        const auto startPosition = list->m_rings.at(0).m_startPosition;
        const auto endPosition = list->m_rings.back().m_endPosition;
        for (int i = 0; i < pStep; i++) {
            archetype.m_position =
                    list->m_rings.at(0).GetPoint(newNormalDir, angleStep * i, true);

            float distanceToStart = 0;
            float distanceToEnd = 1;
            archetype.m_bondId =
                    glm::ivec4(internodeIndex, parentIndices[internodeIndex], -1, -1);
            archetype.m_bondId2 = glm::ivec4(-1, -1, -1, -1);
            archetype.m_weight = glm::vec4(
                    distanceToStart / (distanceToStart + distanceToEnd),
                    distanceToEnd / (distanceToStart + distanceToEnd), 0.0f, 0.0f);
            archetype.m_weight2 = glm::vec4(0.0f);

            const float x =
                    i < pStep / 2 ? i * textureXStep : (pStep - i) * textureXStep;
            archetype.m_texCoords = glm::vec2(x, 0.0f);
            vertices.push_back(archetype);
        }
        std::vector<float> angles;
        angles.resize(step);
        std::vector<float> pAngles;
        pAngles.resize(pStep);

        for (auto i = 0; i < pStep; i++) {
            pAngles[i] = angleStep * i;
        }
        angleStep = 360.0f / static_cast<float>(step);
        for (auto i = 0; i < step; i++) {
            angles[i] = angleStep * i;
        }

        std::vector<unsigned> pTarget;
        std::vector<unsigned> target;
        pTarget.resize(pStep);
        target.resize(step);
        for (int i = 0; i < pStep; i++) {
            // First we allocate nearest vertices for parent.
            auto minAngleDiff = 360.0f;
            for (auto j = 0; j < step; j++) {
                const float diff = glm::abs(pAngles[i] - angles[j]);
                if (diff < minAngleDiff) {
                    minAngleDiff = diff;
                    pTarget[i] = j;
                }
            }
        }
        for (int i = 0; i < step; i++) {
            // Second we allocate nearest vertices for child
            float minAngleDiff = 360.0f;
            for (int j = 0; j < pStep; j++) {
                const float diff = glm::abs(angles[i] - pAngles[j]);
                if (diff < minAngleDiff) {
                    minAngleDiff = diff;
                    target[i] = j;
                }
            }
        }
        for (int i = 0; i < pStep; i++) {
            if (pTarget[i] == pTarget[i == pStep - 1 ? 0 : i + 1]) {
                indices.push_back(vertexIndex + i);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
                indices.push_back(vertexIndex + pStep + pTarget[i]);
            } else {
                indices.push_back(vertexIndex + i);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
                indices.push_back(vertexIndex + pStep + pTarget[i]);

                indices.push_back(vertexIndex + pStep +
                                  pTarget[i == pStep - 1 ? 0 : i + 1]);
                indices.push_back(vertexIndex + pStep + pTarget[i]);
                indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
            }
        }

        vertexIndex += pStep;
        textureXStep = 1.0f / step * 4.0f;
        const int ringSize = list->m_rings.size();
        for (auto ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            for (auto i = 0; i < step; i++) {
                archetype.m_position = list->m_rings.at(ringIndex).GetPoint(
                        newNormalDir, angleStep * i, false);

                float distanceToStart = glm::distance(
                        list->m_rings.at(ringIndex).m_endPosition, startPosition);
                float distanceToEnd = glm::distance(
                        list->m_rings.at(ringIndex).m_endPosition, endPosition);
                archetype.m_bondId =
                        glm::ivec4(internodeIndex, parentIndices[internodeIndex], -1, -1);
                archetype.m_bondId2 = glm::ivec4(-1, -1, -1, -1);
                archetype.m_weight = glm::vec4(
                        distanceToStart / (distanceToStart + distanceToEnd),
                        distanceToEnd / (distanceToStart + distanceToEnd), 0.0f, 0.0f);
                archetype.m_weight2 = glm::vec4(0.0f);

                const auto x =
                        i < (step / 2) ? i * textureXStep : (step - i) * textureXStep;
                const auto y = ringIndex % 2 == 0 ? 1.0f : 0.0f;
                archetype.m_texCoords = glm::vec2(x, y);
                vertices.push_back(archetype);
            }
            if (ringIndex != 0) {
                for (int i = 0; i < step - 1; i++) {
                    // Down triangle
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i);
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
                    indices.push_back(vertexIndex + (ringIndex) * step + i);
                    // Up triangle
                    indices.push_back(vertexIndex + (ringIndex) * step + i + 1);
                    indices.push_back(vertexIndex + (ringIndex) * step + i);
                    indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
                }
                // Down triangle
                indices.push_back(vertexIndex + (ringIndex - 1) * step + step - 1);
                indices.push_back(vertexIndex + (ringIndex - 1) * step);
                indices.push_back(vertexIndex + (ringIndex) * step + step - 1);
                // Up triangle
                indices.push_back(vertexIndex + (ringIndex) * step);
                indices.push_back(vertexIndex + (ringIndex) * step + step - 1);
                indices.push_back(vertexIndex + (ringIndex - 1) * step);
            }
        }
    }
}

