//
// Created by lllll on 8/27/2021.
//

#include "InternodeModel/IPlantBehaviour.hpp"
#include "InternodeModel/Internode.hpp"
#include "Curve.hpp"
#include "InternodeLayer.hpp"
#include "IInternodeFoliage.hpp"
#include "DataComponents.hpp"
#include "TransformLayer.hpp"
#include "Graphics.hpp"

using namespace PlantArchitect;

void IPlantBehaviour::UpdateBranches(const std::shared_ptr<Scene> &scene) {
    std::vector<Entity> roots;
    scene->GetEntityArray(m_rootsQuery, roots);
    for (const auto &root: roots) {
        if (!RootCheck(scene, root)) return;
        Entity rootInternode, rootBranch;
        scene->ForEachChild(root, [&](Entity child) {
            if (InternodeCheck(scene, child)) rootInternode = child;
            else if (BranchCheck(scene, child)) rootBranch = child;
        });
        if (!scene->IsEntityValid(rootInternode) || !scene->IsEntityValid(rootBranch)) return;
        auto children = scene->GetChildren(rootBranch);
        for (const auto &i: children) scene->DeleteEntity(i);
        auto branch = scene->GetOrSetPrivateComponent<Branch>(rootBranch).lock();
        branch->m_internodeChain.clear();
        BranchGraphWalkerRootToEnd(scene, rootBranch, [&](Entity parent, Entity child) {
            scene->GetOrSetPrivateComponent<Branch>(child).lock()->m_internodeChain.clear();
        });
        UpdateBranchHelper(scene, rootBranch, rootInternode);
        {
            auto branchInfo = scene->GetDataComponent<InternodeBranchInfo>(rootBranch);
            auto branchStartInternodeGT = scene->GetDataComponent<GlobalTransform>(branch->m_internodeChain.front());
            auto branchEndInternodeGT = scene->GetDataComponent<GlobalTransform>(branch->m_internodeChain.back());
            auto branchStartInternodeInfo = scene->GetDataComponent<InternodeInfo>(branch->m_internodeChain.front());
            auto branchEndInternodeInfo = scene->GetDataComponent<InternodeInfo>(branch->m_internodeChain.back());
            glm::vec3 scale;
            glm::quat rotation;
            glm::vec3 translation;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(branchEndInternodeGT.m_value, scale, rotation, translation, skew,
                           perspective);
            auto branchEndPosition = translation + rotation * glm::vec3(0, 0, -1) * branchEndInternodeInfo.m_length;
            auto branchStartPosition = branchStartInternodeGT.GetPosition();
            GlobalTransform branchGT;
            branchInfo.m_length = glm::length(branchStartPosition - branchEndPosition);
            if (branchInfo.m_length > 0)
                branchGT.m_value = glm::translate(branchEndPosition) * glm::mat4_cast(
                        glm::quatLookAt(glm::normalize(branchEndPosition - branchStartPosition),
                                        rotation * glm::vec3(0, 1, 0))) * glm::scale(glm::vec3(1.0f));
            else branchGT.SetPosition(branchEndPosition);
            branchInfo.m_thickness = branchStartInternodeInfo.m_thickness;

            scene->SetDataComponent(rootBranch, branchGT);
            scene->SetDataComponent(rootBranch, branchInfo);
        }
        BranchGraphWalkerRootToEnd(scene, rootBranch, [&](Entity parent, Entity child) {
            auto branch = scene->GetOrSetPrivateComponent<Branch>(child).lock();
            auto branchInfo = scene->GetDataComponent<InternodeBranchInfo>(child);
            auto branchStartInternodeGT = scene->GetDataComponent<GlobalTransform>(branch->m_internodeChain.front());
            auto branchEndInternodeGT = scene->GetDataComponent<GlobalTransform>(branch->m_internodeChain.back());
            auto branchStartInternodeInfo = scene->GetDataComponent<InternodeInfo>(branch->m_internodeChain.front());
            auto branchEndInternodeInfo = scene->GetDataComponent<InternodeInfo>(branch->m_internodeChain.back());
            glm::vec3 scale;
            glm::quat rotation;
            glm::vec3 translation;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(branchEndInternodeGT.m_value, scale, rotation, translation, skew,
                           perspective);
            auto branchEndPosition = translation + rotation * glm::vec3(0, 0, -1) * branchEndInternodeInfo.m_length;
            auto branchStartPosition = branchStartInternodeGT.GetPosition();
            GlobalTransform branchGT;
            branchInfo.m_length = glm::length(branchStartPosition - branchEndPosition);
            if (branchInfo.m_length > 0)
                branchGT.m_value = glm::translate(branchEndPosition) * glm::mat4_cast(
                        glm::quatLookAt(glm::normalize(branchEndPosition - branchStartPosition),
                                        rotation * glm::vec3(0, 1, 0))) * glm::scale(glm::vec3(1.0f));
            else branchGT.SetPosition(branchEndPosition);

            branchInfo.m_thickness = branchStartInternodeInfo.m_thickness;

            scene->SetDataComponent(child, branchGT);
            scene->SetDataComponent(child, branchInfo);
        });
        Application::GetLayer<TransformLayer>()->CalculateTransformGraphForDescendents(
                scene,
                rootBranch);
    }
}

void IPlantBehaviour::DestroyInternode(const std::shared_ptr<Scene> &scene, const Entity &internode) {
    std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
    scene->DeleteEntity(internode);
}

void IPlantBehaviour::DestroyBranch(const std::shared_ptr<Scene> &scene, const Entity &branch) {
    std::lock_guard<std::mutex> lockGuard(m_branchFactoryLock);
    auto internode = scene->GetOrSetPrivateComponent<Branch>(branch).lock()->m_currentInternode.Get();
    if (InternodeCheck(scene, internode))DestroyInternode(scene, internode);
    scene->DeleteEntity(branch);
}

void
IPlantBehaviour::GenerateSkinnedMeshes(const std::shared_ptr<Scene> &scene, const MeshGeneratorSettings &settings) {
    UpdateBranches(scene);
    std::vector<Entity> currentRoots;
    scene->GetEntityArray(m_rootsQuery, currentRoots);
    int plantSize = currentRoots.size();
    std::vector<std::vector<Entity>> boundEntitiesLists;
    std::vector<std::vector<unsigned>> boneIndicesLists;
    std::vector<std::vector<int>> parentIndicesLists;
    boundEntitiesLists.resize(plantSize);
    boneIndicesLists.resize(plantSize);
    parentIndicesLists.resize(plantSize);

    //Use internal JobSystem to dispatch job for entity collection.
    std::vector<std::shared_future<void>> results;
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        results.push_back(Jobs::Workers().Push([&, plantIndex](int id) {
            auto children = scene->GetChildren(currentRoots[plantIndex]);
            for (const auto &child: children) {
                if (BranchCheck(scene, child)) {
                    BranchCollector(scene, boundEntitiesLists[plantIndex],
                                    parentIndicesLists[plantIndex], -1, child);
                    break;
                }
            }
        }).share());
    }
    for (const auto &i: results)
        i.wait();

#pragma region Prepare rings for branch mesh.
    if (settings.m_enableBranch) {
        PrepareBranchRings(scene, settings);
    }
#pragma endregion
#pragma region Prepare foliage transforms.
    if (settings.m_enableFoliage) {
        PrepareFoliageMatrices(scene, settings);
    }
#pragma endregion

    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        const auto &rootEntity = currentRoots[plantIndex];
        auto children = scene->GetChildren(rootEntity);
        Entity rootInternode, rootBranch;
        for (const auto &child: children) {
            if (InternodeCheck(scene, child)) rootInternode = child;
            else if (BranchCheck(scene, child)) rootBranch = child;
        }
        Entity branchMesh, foliageMesh;
        PrepareInternodeForSkeletalAnimation(scene, rootEntity, branchMesh, foliageMesh, settings);
        if (settings.m_enableBranch) {
#pragma region Branch mesh
            auto animator = scene->GetOrSetPrivateComponent<Animator>(branchMesh).lock();
            auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(branchMesh).lock();
            auto material = skinnedMeshRenderer->m_material.Get<Material>();
            skinnedMeshRenderer->SetEnabled(true);
            auto root = scene->GetOrSetPrivateComponent<InternodePlant>(rootEntity).lock();
            auto texture = root->m_plantDescriptor.Get<IPlantDescriptor>()->m_branchTexture.Get<Texture2D>();
            if (texture)
                material->m_albedoTexture = texture;
            material->m_materialProperties.m_albedoColor = root->m_plantDescriptor.Get<IPlantDescriptor>()->m_branchColor;
            material->m_vertexColorOnly = settings.m_vertexColorOnly;
            auto internode = scene->GetOrSetPrivateComponent<Internode>(rootInternode).lock();
            const auto plantGlobalTransform =
                    scene->GetDataComponent<GlobalTransform>(rootEntity);
#pragma region Animator
            std::vector<glm::mat4> offsetMatrices;
            std::vector<std::string> names;
            offsetMatrices.resize(boundEntitiesLists[plantIndex].size());
            names.resize(boundEntitiesLists[plantIndex].size());
            boneIndicesLists[plantIndex].resize(
                    boundEntitiesLists[plantIndex].size());
            for (int i = 0; i < boundEntitiesLists[plantIndex].size(); i++) {
                names[i] = scene->GetEntityName(boundEntitiesLists[plantIndex][i]);
                offsetMatrices[i] =
                        glm::inverse(glm::inverse(plantGlobalTransform.m_value) *
                                     scene->GetDataComponent<GlobalTransform>(boundEntitiesLists[plantIndex][i])
                                             .m_value);
                boneIndicesLists[plantIndex][i] = i;
            }
            animator->Setup(names, offsetMatrices);
            skinnedMeshRenderer->SetRagDoll(true);
            skinnedMeshRenderer->SetRagDollBoundEntities(boundEntitiesLists[plantIndex], false);
#pragma endregion
            std::vector<unsigned> skinnedIndices;
            std::vector<SkinnedVertex> skinnedVertices;
            BranchSkinnedMeshGenerator(scene, boundEntitiesLists[plantIndex],
                                       parentIndicesLists[plantIndex],
                                       skinnedVertices, skinnedIndices, settings);
            if (!skinnedVertices.empty()) {
                auto skinnedMesh = ProjectManager::CreateTemporaryAsset<SkinnedMesh>();
                skinnedMesh->SetVertices(
                        17, skinnedVertices, skinnedIndices);
                skinnedMesh
                        ->m_boneAnimatorIndices = boneIndicesLists[plantIndex];
                skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(skinnedMesh);
                skinnedMeshRenderer->GetBoneMatrices();
                skinnedMeshRenderer->m_ragDollFreeze = true;
            }
#pragma endregion
        }
        if (settings.m_enableFoliage) {
#pragma region Foliage mesh
            auto animator = scene->GetOrSetPrivateComponent<Animator>(foliageMesh).lock();
            auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(foliageMesh).lock();
            auto material = skinnedMeshRenderer->m_material.Get<Material>();
            skinnedMeshRenderer->SetEnabled(true);
            auto root = scene->GetOrSetPrivateComponent<InternodePlant>(rootEntity).lock();
            auto foliageModule = root->m_plantDescriptor.Get<IPlantDescriptor>()->m_foliagePhyllotaxis.Get<IInternodeFoliage>();
            if (foliageModule) {
                auto texture = foliageModule->m_foliageTexture.Get<Texture2D>();
                if (texture)
                    material->m_albedoTexture = texture;
            }
            material->m_materialProperties.m_albedoColor = root->m_plantDescriptor.Get<IPlantDescriptor>()->m_foliageColor;
            material->m_vertexColorOnly = settings.m_vertexColorOnly;
            const auto plantGlobalTransform =
                    scene->GetDataComponent<GlobalTransform>(rootEntity);
#pragma region Animator
            std::vector<glm::mat4> offsetMatrices;
            std::vector<std::string> names;
            offsetMatrices.resize(boundEntitiesLists[plantIndex].size());
            names.resize(boundEntitiesLists[plantIndex].size());
            boneIndicesLists[plantIndex].resize(
                    boundEntitiesLists[plantIndex].size());
            for (int i = 0; i < boundEntitiesLists[plantIndex].size(); i++) {
                names[i] = scene->GetEntityName(boundEntitiesLists[plantIndex][i]);
                offsetMatrices[i] =
                        glm::inverse(glm::inverse(plantGlobalTransform.m_value) *
                                     scene->GetDataComponent<GlobalTransform>(boundEntitiesLists[plantIndex][i])
                                             .m_value);
                boneIndicesLists[plantIndex][i] = i;
            }
            animator->Setup(names, offsetMatrices);
            skinnedMeshRenderer->SetRagDoll(true);
            skinnedMeshRenderer->SetRagDollBoundEntities(boundEntitiesLists[plantIndex], false);
#pragma endregion
            std::vector<unsigned> skinnedIndices;
            std::vector<SkinnedVertex> skinnedVertices;
            FoliageSkinnedMeshGenerator(scene, boundEntitiesLists[plantIndex],
                                        parentIndicesLists[plantIndex],
                                        skinnedVertices, skinnedIndices, settings);
            if (!skinnedVertices.empty()) {
                auto skinnedMesh = ProjectManager::CreateTemporaryAsset<SkinnedMesh>();
                skinnedMesh->SetVertices(
                        17, skinnedVertices, skinnedIndices);
                skinnedMesh
                        ->m_boneAnimatorIndices = boneIndicesLists[plantIndex];
                skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(skinnedMesh);
                skinnedMeshRenderer->GetBoneMatrices();
                skinnedMeshRenderer->m_ragDollFreeze = true;
            }
#pragma endregion
        }
    }
}

void IPlantBehaviour::BranchCollector(const std::shared_ptr<Scene> &scene, std::vector<Entity> &boundEntities,
                                      std::vector<int> &parentIndices,
                                      const int &parentIndex, const Entity &node) {
    if (!BranchCheck(scene, node)) return;
    boundEntities.push_back(node);
    parentIndices.push_back(parentIndex);
    const size_t currentIndex = boundEntities.size() - 1;
    scene->ForEachChild(node, [&](Entity child) {
        if (BranchCheck(scene, child)) BranchCollector(scene, boundEntities, parentIndices, currentIndex, child);
    });

}

void IPlantBehaviour::FoliageSkinnedMeshGenerator(const std::shared_ptr<Scene> &scene, std::vector<Entity> &entities,
                                                  std::vector<int> &parentIndices,
                                                  std::vector<SkinnedVertex> &vertices,
                                                  std::vector<unsigned int> &indices,
                                                  const MeshGeneratorSettings &settings) {
    auto quadMesh = DefaultResources::Primitives::Quad;
    auto &quadTriangles = quadMesh->UnsafeGetTriangles();
    auto quadVerticesSize = quadMesh->GetVerticesAmount();
    size_t offset = 0;
    for (int branchIndex = 0; branchIndex < entities.size();
         branchIndex++) {
        int parentIndex = 0;
        if (branchIndex != 0) parentIndex = parentIndices[branchIndex];
        auto &branchEntity = entities[branchIndex];
        auto branchGlobalTransform = scene->GetDataComponent<GlobalTransform>(branchEntity);
        auto branch = scene->GetOrSetPrivateComponent<Branch>(branchEntity).lock();
        for (const auto &internodeEntity: branch->m_internodeChain) {
            auto internodeGlobalTransform = scene->GetDataComponent<GlobalTransform>(internodeEntity);
            auto internode = scene->GetOrSetPrivateComponent<Internode>(internodeEntity).lock();
            glm::vec3 newNormalDir;
            if (branchIndex != 0) {
                newNormalDir = scene->GetOrSetPrivateComponent<Internode>(entities[parentIndex]).lock()->m_normalDir;
            } else {
                newNormalDir = internodeGlobalTransform.GetRotation() *
                               glm::vec3(1.0f, 0.0f, 0.0f);
            }
            const glm::vec3 front =
                    internodeGlobalTransform.GetRotation() *
                    glm::vec3(0.0f, 0.0f, -1.0f);
            newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);
            internode->m_normalDir = newNormalDir;
            for (const auto &matrix: internode->m_foliageMatrices) {
                SkinnedVertex archetype;
                if (settings.m_overrideVertexColor) archetype.m_color = settings.m_foliageVertexColor;
                else archetype.m_color = glm::vec4(1.0f);
                for (auto i = 0; i < quadMesh->GetVerticesAmount(); i++) {
                    archetype.m_position =
                            matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_position, 1.0f);
                    archetype.m_normal = glm::normalize(glm::vec3(
                            matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_normal, 0.0f)));
                    archetype.m_tangent = glm::normalize(glm::vec3(
                            matrix *
                            glm::vec4(quadMesh->UnsafeGetVertices()[i].m_tangent, 0.0f)));
                    archetype.m_texCoord =
                            quadMesh->UnsafeGetVertices()[i].m_texCoord;
                    archetype.m_bondId = glm::ivec4(branchIndex, -1, -1, -1);
                    archetype.m_weight = glm::vec4(1, 0, 0, 0);
                    archetype.m_bondId2 = glm::ivec4(-1, -1, -1, -1);
                    archetype.m_weight2 = glm::vec4(0, 0, 0, 0);
                    vertices.push_back(archetype);
                }
                for (auto triangle: quadTriangles) {
                    triangle.x += offset;
                    triangle.y += offset;
                    triangle.z += offset;
                    indices.push_back(triangle.x);
                    indices.push_back(triangle.y);
                    indices.push_back(triangle.z);
                }
                offset += quadVerticesSize;
            }
        }
    }
}

void IPlantBehaviour::BranchSkinnedMeshGenerator(const std::shared_ptr<Scene> &scene, std::vector<Entity> &entities,
                                                 std::vector<int> &parentIndices,
                                                 std::vector<SkinnedVertex> &vertices,
                                                 std::vector<unsigned int> &indices,
                                                 const MeshGeneratorSettings &settings) {
    int parentStep = -1;
    for (int branchIndex = 0; branchIndex < entities.size();
         branchIndex++) {
        int parentIndex = 0;
        if (branchIndex != 0) parentIndex = parentIndices[branchIndex];
        auto &branchEntity = entities[branchIndex];
        auto branchGlobalTransform = scene->GetDataComponent<GlobalTransform>(branchEntity);
        auto branch = scene->GetOrSetPrivateComponent<Branch>(branchEntity).lock();
        for (const auto &internodeEntity: branch->m_internodeChain) {
            bool isOnlyChild = scene->GetChildrenAmount(scene->GetParent(internodeEntity)) == 1;
            bool hasMultipleChild = scene->GetChildrenAmount(internodeEntity) > 1;
            auto internodeGlobalTransform = scene->GetDataComponent<GlobalTransform>(internodeEntity);
            glm::vec3 newNormalDir;
            if (branchIndex != 0) {
                newNormalDir = scene->GetOrSetPrivateComponent<Internode>(entities[parentIndex]).lock()->m_normalDir;
            } else {
                newNormalDir = internodeGlobalTransform.GetRotation() *
                               glm::vec3(1.0f, 0.0f, 0.0f);
            }
            if (scene->HasDataComponent<InternodeRootInfo>(scene->GetParent(internodeEntity))) {
                isOnlyChild = true;
                hasMultipleChild = false;
            }
            bool markJunction = false;
            if (settings.m_markJunctions) {
                markJunction = true;
                if (isOnlyChild) {
                    auto parent = scene->GetParent(internodeEntity);
                    scene->ForEachChild(parent, [&](Entity child) {
                        if (scene->HasDataComponent<InternodeInfo>(child) &&
                            scene->GetDataComponent<InternodeInfo>(child).m_length < 0.8f) {
                            markJunction = false;
                        }
                    });
                } else if (hasMultipleChild) {
                    scene->ForEachChild(internodeEntity, [&](Entity child) {
                        if (scene->HasDataComponent<InternodeInfo>(child) &&
                            scene->GetDataComponent<InternodeInfo>(child).m_length < 0.8f) {
                            markJunction = false;
                        }
                    });
                }
            }
            const glm::vec3 front =
                    internodeGlobalTransform.GetRotation() *
                    glm::vec3(0.0f, 0.0f, -1.0f);
            newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);
            auto internode = scene->GetOrSetPrivateComponent<Internode>(internodeEntity).lock();
            auto branchColor = scene->GetDataComponent<InternodeColor>(internodeEntity);
            internode->m_normalDir = newNormalDir;
            if (internode->m_rings.empty()) {
                continue;
            }
            auto step = internode->m_step;
            // For stitching
            int pStep = parentStep > 0 ? parentStep : step;
            parentStep = step;
            pStep = step;
            float angleStep = 360.0f / static_cast<float>(pStep);
            int vertexIndex = vertices.size();
            SkinnedVertex archetype;

            float textureXStep = 1.0f / pStep * 4.0f;

            const auto startPosition = internode->m_rings.at(0).m_startPosition;
            const auto endPosition = internode->m_rings.back().m_endPosition;
            for (int i = 0; i < pStep; i++) {
                archetype.m_position =
                        internode->m_rings.at(0).GetPoint(newNormalDir, angleStep * i, true);
                float distanceToStart = 0;
                float distanceToEnd = 1;
                archetype.m_bondId =
                        glm::ivec4(branchIndex, parentIndex, -1, -1);
                archetype.m_bondId2 = glm::ivec4(-1, -1, -1, -1);
                archetype.m_weight = glm::vec4(
                        distanceToStart / (distanceToStart + distanceToEnd),
                        distanceToEnd / (distanceToStart + distanceToEnd), 0.0f, 0.0f);
                archetype.m_weight2 = glm::vec4(0.0f);

                const float x =
                        i < pStep / 2 ? i * textureXStep : (pStep - i) * textureXStep;
                archetype.m_texCoord = glm::vec2(x, 0.0f);
                if (markJunction) {
                    archetype.m_color = glm::normalize(internode->m_rings.at(0).m_startAxis);
                    if (!isOnlyChild) {
                        archetype.m_color *= scene->GetParent(internodeEntity).GetIndex();
                    } else {
                        archetype.m_color *= 0.5f;
                    }
                } else if (settings.m_overrideVertexColor) archetype.m_color = settings.m_branchVertexColor;
                else archetype.m_color = branchColor.m_value;
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
            const int ringSize = internode->m_rings.size();
            for (auto ringIndex = 0; ringIndex < ringSize; ringIndex++) {
                for (auto i = 0; i < step; i++) {
                    archetype.m_position = internode->m_rings.at(ringIndex).GetPoint(
                            newNormalDir, angleStep * i, false);
                    float distanceToStart = glm::distance(
                            internode->m_rings.at(ringIndex).m_endPosition, startPosition);
                    float distanceToEnd = glm::distance(
                            internode->m_rings.at(ringIndex).m_endPosition, endPosition);
                    archetype.m_bondId =
                            glm::ivec4(branchIndex, parentIndex, -1, -1);
                    archetype.m_bondId2 = glm::ivec4(-1, -1, -1, -1);
                    archetype.m_weight = glm::vec4(
                            distanceToStart / (distanceToStart + distanceToEnd),
                            distanceToEnd / (distanceToStart + distanceToEnd), 0.0f, 0.0f);
                    archetype.m_weight2 = glm::vec4(0.0f);

                    const auto x =
                            i < (step / 2) ? i * textureXStep : (step - i) * textureXStep;
                    const auto y = ringIndex % 2 == 0 ? 1.0f : 0.0f;
                    archetype.m_texCoord = glm::vec2(x, y);
                    auto ratio = (float) ringIndex / (ringSize - 1);
                    if (markJunction) {
                        archetype.m_color = glm::normalize(internode->m_rings.at(ringIndex).m_endAxis);
                        if (ratio <= settings.m_junctionLowerRatio && !isOnlyChild) {
                            archetype.m_color *= scene->GetParent(internodeEntity).GetIndex();
                        } else if (ratio >= 1.0f - settings.m_junctionUpperRatio && hasMultipleChild) {
                            archetype.m_color *= internodeEntity.GetIndex();
                        } else {
                            archetype.m_color *= 0.5f;
                        }
                    } else if (settings.m_overrideVertexColor) archetype.m_color = settings.m_branchVertexColor;
                    else archetype.m_color = branchColor.m_value;
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
}

void MeshGeneratorSettings::OnInspect() {
    if (ImGui::TreeNodeEx("Mesh Generator settings")) {
        ImGui::DragFloat("Resolution", &m_resolution, 0.001f);
        ImGui::DragFloat("Subdivision", &m_subdivision, 0.001f);
        ImGui::Checkbox("Vertex color only", &m_vertexColorOnly);
        ImGui::Checkbox("Foliage", &m_enableFoliage);
        ImGui::Checkbox("Branch", &m_enableBranch);
        ImGui::Checkbox("Smoothness", &m_smoothness);
        if (!m_smoothness) {
            ImGui::DragFloat("Internode length factor", &m_internodeLengthFactor, 0.001f, 0.0f, 1.0f);
        }
        ImGui::Checkbox("Override radius", &m_overrideRadius);
        if (m_overrideRadius) ImGui::DragFloat("Radius", &m_radius);
        ImGui::Checkbox("Override vertex color", &m_overrideVertexColor);
        if (m_overrideVertexColor) {
            ImGui::ColorEdit3("Branch vertex color", &m_branchVertexColor.x);
            ImGui::ColorEdit3("Foliage vertex color", &m_foliageVertexColor.x);
        }
        ImGui::Checkbox("Mark Junctions", &m_markJunctions);
        if (m_markJunctions) {
            ImGui::DragFloat("Junction Lower Ratio", &m_junctionLowerRatio, 0.01f, 0.0f, 0.5f);
            ImGui::DragFloat("Junction Upper Ratio", &m_junctionUpperRatio, 0.01f, 0.0f, 0.5f);
        }
        ImGui::TreePop();
    }
}

void SubtreeSettings::OnInspect() {
    if (ImGui::TreeNodeEx("Subtree settings")) {
        ImGui::DragInt("Layer", &m_layer);
        ImGui::Checkbox("Include base internode", &m_enableBaseInternode);
        ImGui::DragFloat("Resolution", &m_resolution, 0.001f);
        ImGui::DragFloat("Subdivision", &m_subdivision, 0.001f);
        ImGui::Checkbox("Base", &m_enableBase);
        ImGui::Checkbox("Line", &m_enableLines);
        ImGui::Checkbox("Point", &m_enablePoints);
        ImGui::Checkbox("Arrow", &m_enableArrows);


        ImGui::DragFloat("Line radius", &m_lineRadius, 0.001f);
        ImGui::Checkbox("Line smoothness", &m_lineSmoothness);
        ImGui::DragFloat("Line length factor", &m_lineLengthFactor, 0.01f);
        if (m_enablePoints) ImGui::ColorEdit3("Point color", &m_pointColor.x);
        if (m_enableLines) ImGui::ColorEdit3("Line color", &m_lineColor.x);
        if (m_enableArrows) ImGui::ColorEdit3("Arrow color", &m_arrowColor.x);
    }
}

void MeshGeneratorSettings::Save(const std::string &name, YAML::Emitter &out) {
    out << YAML::Key << name << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "m_resolution" << YAML::Value << m_resolution;
    out << YAML::Key << "m_subdivision" << YAML::Value << m_subdivision;
    out << YAML::Key << "m_vertexColorOnly" << YAML::Value << m_vertexColorOnly;
    out << YAML::Key << "m_enableFoliage" << YAML::Value << m_enableFoliage;
    out << YAML::Key << "m_enableBranch" << YAML::Value << m_enableBranch;
    out << YAML::Key << "m_smoothness" << YAML::Value << m_smoothness;
    out << YAML::Key << "m_overrideRadius" << YAML::Value << m_overrideRadius;
    out << YAML::Key << "m_boundaryRadius" << YAML::Value << m_radius;
    out << YAML::Key << "m_internodeLengthFactor" << YAML::Value << m_internodeLengthFactor;
    out << YAML::Key << "m_overrideVertexColor" << YAML::Value << m_overrideVertexColor;
    out << YAML::Key << "m_markJunctions" << YAML::Value << m_markJunctions;
    out << YAML::Key << "m_junctionUpperRatio" << YAML::Value << m_junctionUpperRatio;
    out << YAML::Key << "m_junctionLowerRatio" << YAML::Value << m_junctionLowerRatio;
    out << YAML::Key << "m_branchVertexColor" << YAML::Value << m_branchVertexColor;
    out << YAML::Key << "m_foliageVertexColor" << YAML::Value << m_foliageVertexColor;
    out << YAML::EndMap;
}

void MeshGeneratorSettings::Load(const std::string &name, const YAML::Node &in) {
    if (in[name]) {
        const auto &ms = in[name];
        if (ms["m_resolution"]) m_resolution = ms["m_resolution"].as<float>();
        if (ms["m_subdivision"]) m_subdivision = ms["m_subdivision"].as<float>();
        if (ms["m_vertexColorOnly"]) m_vertexColorOnly = ms["m_vertexColorOnly"].as<bool>();
        if (ms["m_enableFoliage"]) m_enableFoliage = ms["m_enableFoliage"].as<bool>();
        if (ms["m_enableBranch"]) m_enableBranch = ms["m_enableBranch"].as<bool>();
        if (ms["m_smoothness"]) m_smoothness = ms["m_smoothness"].as<bool>();
        if (ms["m_overrideRadius"]) m_overrideRadius = ms["m_overrideRadius"].as<bool>();
        if (ms["m_boundaryRadius"]) m_radius = ms["m_boundaryRadius"].as<float>();
        if (ms["m_internodeLengthFactor"]) m_internodeLengthFactor = ms["m_internodeLengthFactor"].as<float>();
        if (ms["m_overrideVertexColor"]) m_overrideVertexColor = ms["m_overrideVertexColor"].as<bool>();
        if (ms["m_markJunctions"]) m_markJunctions = ms["m_markJunctions"].as<bool>();
        if (ms["m_junctionUpperRatio"]) m_junctionUpperRatio = ms["m_junctionUpperRatio"].as<float>();
        if (ms["m_junctionLowerRatio"]) m_junctionLowerRatio = ms["m_junctionLowerRatio"].as<float>();
        if (ms["m_branchVertexColor"]) m_branchVertexColor = ms["m_branchVertexColor"].as<glm::vec3>();
        if (ms["m_foliageVertexColor"]) m_foliageVertexColor = ms["m_foliageVertexColor"].as<glm::vec3>();
    }
}

void
IPlantBehaviour::PrepareInternodeForSkeletalAnimation(const std::shared_ptr<Scene> &scene, const Entity &entity,
                                                      Entity &branchMesh, Entity &foliageMesh,
                                                      const MeshGeneratorSettings &settings) {
    auto children = scene->GetChildren(entity);
    for (const auto &child: children) {
        if (scene->GetEntityName(child) == "BranchMesh" || scene->GetEntityName(child) == "FoliageMesh") {
            scene->DeleteEntity(child);
        }
    }

    if (settings.m_enableBranch) {
        if (branchMesh.GetIndex() == 0) branchMesh = scene->CreateEntity("BranchMesh");
        auto animator = scene->GetOrSetPrivateComponent<Animator>(branchMesh).lock();
        auto skinnedMeshRenderer =
                scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(branchMesh).lock();
        skinnedMeshRenderer->m_skinnedMesh = ProjectManager::CreateTemporaryAsset<SkinnedMesh>();
        auto skinnedMat = ProjectManager::CreateTemporaryAsset<Material>();
        skinnedMat->SetProgram(DefaultResources::GLPrograms::StandardSkinnedProgram);
        skinnedMeshRenderer->m_material = skinnedMat;
        skinnedMat->m_materialProperties.m_albedoColor = glm::vec3(40.0f / 255, 15.0f / 255, 0.0f);
        skinnedMat->m_materialProperties.m_roughness = 1.0f;
        skinnedMat->m_materialProperties.m_metallic = 0.0f;
        skinnedMeshRenderer->m_animator = scene->GetOrSetPrivateComponent<Animator>(branchMesh).lock();
        scene->SetParent(branchMesh, entity);
    }
    if (settings.m_enableFoliage) {
        if (foliageMesh.GetIndex() == 0) foliageMesh = scene->CreateEntity("FoliageMesh");
        auto animator = scene->GetOrSetPrivateComponent<Animator>(foliageMesh).lock();
        auto skinnedMeshRenderer =
                scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(foliageMesh).lock();
        skinnedMeshRenderer->m_skinnedMesh = ProjectManager::CreateTemporaryAsset<SkinnedMesh>();
        auto skinnedMat = ProjectManager::CreateTemporaryAsset<Material>();
        skinnedMat->SetProgram(DefaultResources::GLPrograms::StandardSkinnedProgram);
        skinnedMeshRenderer->m_material = skinnedMat;
        skinnedMat->m_materialProperties.m_albedoColor = glm::vec3(0.0f, 1.0f, 0.0f);
        skinnedMat->m_materialProperties.m_roughness = 1.0f;
        skinnedMat->m_materialProperties.m_metallic = 0.0f;
        skinnedMat->m_drawSettings.m_cullFace = false;
        skinnedMeshRenderer->m_animator = scene->GetOrSetPrivateComponent<Animator>(foliageMesh).lock();
        scene->SetParent(foliageMesh, entity);
    }
}

Entity IPlantBehaviour::CreateBranchHelper(const std::shared_ptr<Scene> &scene, const Entity &parent,
                                           const Entity &internode) {
    Entity retVal;
    std::lock_guard<std::mutex> lockGuard(m_branchFactoryLock);
    retVal = scene->CreateEntity(m_branchArchetype, "Branch");
    scene->SetParent(retVal, parent);
    InternodeBranchInfo branchInfo;
    scene->SetDataComponent(retVal, branchInfo);
    auto parentBranch = scene->GetOrSetPrivateComponent<Branch>(parent).lock();
    auto branch = scene->GetOrSetPrivateComponent<Branch>(retVal).lock();
    branch->m_currentRoot = parentBranch->m_currentRoot;
    branch->m_currentInternode = internode;
    return retVal;
}

void
IPlantBehaviour::InternodeGraphWalker(const std::shared_ptr<Scene> &scene, const Entity &startInternode,
                                      const std::function<void(Entity, Entity)> &rootToEndAction,
                                      const std::function<void(Entity)> &endToRootAction,
                                      const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startInternode;
    while (scene->GetChildrenAmount(currentNode) == 1) {
        Entity child = scene->GetChildren(currentNode)[0];
        if (InternodeCheck(scene, child)) {
            rootToEndAction(currentNode, child);
            if (InternodeCheck(scene, child)) {
                currentNode = child;
            }
        }
    }
    int trueChildAmount = 0;
    if (scene->GetChildrenAmount(currentNode) != 0) {
        auto children = scene->GetChildren(currentNode);
        for (const auto &child: children) {
            if (InternodeCheck(scene, child)) {
                rootToEndAction(currentNode, child);
                if (InternodeCheck(scene, child)) {
                    InternodeGraphWalker(scene, child, rootToEndAction, endToRootAction, endNodeAction);
                    if (InternodeCheck(scene, child)) {
                        trueChildAmount++;
                    }
                }
            }
        }
    }
    if (trueChildAmount == 0) {
        endNodeAction(currentNode);
    } else {
        endToRootAction(currentNode);
    }
    while (currentNode != startInternode) {
        auto parent = scene->GetParent(currentNode);
        endToRootAction(parent);
        if (!InternodeCheck(scene, currentNode)) {
            endNodeAction(parent);
        }
        currentNode = parent;
    }
}

bool IPlantBehaviour::InternodeCheck(const std::shared_ptr<Scene> &scene, const Entity &target) {
    return scene->IsEntityValid(target) && scene->IsEntityEnabled(target) &&
           scene->HasDataComponent<InternodeInfo>(target) &&
           scene->HasPrivateComponent<Internode>(target) &&
           InternalInternodeCheck(scene, target);
}

bool IPlantBehaviour::RootCheck(const std::shared_ptr<Scene> &scene, const Entity &target) {
    return scene->IsEntityValid(target) && scene->IsEntityEnabled(target) &&
           scene->HasDataComponent<InternodeRootInfo>(target) &&
           scene->HasPrivateComponent<InternodePlant>(target) &&
           InternalRootCheck(scene, target);
}

bool IPlantBehaviour::BranchCheck(const std::shared_ptr<Scene> &scene, const Entity &target) {
    return scene->IsEntityValid(target) && scene->IsEntityEnabled(target) &&
           scene->HasDataComponent<InternodeBranchInfo>(target) &&
           scene->HasPrivateComponent<Branch>(target) &&
           InternalBranchCheck(scene, target);
}

void IPlantBehaviour::UpdateBranchHelper(const std::shared_ptr<Scene> &scene, const Entity &currentBranch,
                                         const Entity &currentInternode) {
    int trueChildAmount = 0;
    scene->ForEachChild(currentInternode, [&](Entity child) {
        if (InternodeCheck(scene, child)) trueChildAmount++;
    });
    scene->GetOrSetPrivateComponent<Branch>(currentBranch).lock()->m_internodeChain.push_back(currentInternode);
    if (trueChildAmount > 1) {
        InternodeBranchInfo branchInfo;
        branchInfo.m_endNode = false;
        scene->SetDataComponent(currentBranch, branchInfo);
        scene->ForEachChild(currentInternode, [&](Entity child) {
            if (!InternodeCheck(scene, child)) return;
            auto newBranch = CreateBranch(scene, currentBranch, child);
            UpdateBranchHelper(scene, newBranch, child);
        });
    } else if (trueChildAmount == 1) {
        scene->ForEachChild(currentInternode, [&](Entity child) {
            if (!InternodeCheck(scene, child)) return;
            UpdateBranchHelper(scene, currentBranch, child);
        });
    }

}

void IPlantBehaviour::InternodeGraphWalkerRootToEnd(const std::shared_ptr<Scene> &scene, const Entity &startInternode,
                                                    const std::function<void(Entity, Entity)> &rootToEndAction) {
    auto currentNode = startInternode;
    while (scene->GetChildrenAmount(currentNode) == 1) {
        Entity child = scene->GetChildren(currentNode)[0];
        if (InternodeCheck(scene, child)) {
            rootToEndAction(currentNode, child);
            if (InternodeCheck(scene, child)) {
                currentNode = child;
            }
        }
    }
    if (scene->GetChildrenAmount(currentNode) != 0) {
        auto children = scene->GetChildren(currentNode);
        for (const auto &child: children) {
            if (InternodeCheck(scene, child)) {
                rootToEndAction(currentNode, child);
                if (InternodeCheck(scene, child)) {
                    InternodeGraphWalkerRootToEnd(scene, child, rootToEndAction);
                }
            }
        }
    }
}

void IPlantBehaviour::InternodeGraphWalkerEndToRoot(const std::shared_ptr<Scene> &scene, const Entity &startInternode,
                                                    const std::function<void(Entity)> &endToRootAction,
                                                    const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startInternode;
    while (scene->GetChildrenAmount(currentNode) == 1) {
        Entity child = scene->GetChildren(currentNode)[0];
        if (InternodeCheck(scene, child)) {
            currentNode = child;
        }
    }
    int trueChildAmount = 0;
    if (scene->GetChildrenAmount(currentNode) != 0) {
        auto children = scene->GetChildren(currentNode);
        for (const auto &child: children) {
            if (InternodeCheck(scene, child)) {
                InternodeGraphWalkerEndToRoot(scene, child, endToRootAction, endNodeAction);
                if (InternodeCheck(scene, child)) {
                    trueChildAmount++;
                }
            }
        }
    }
    if (trueChildAmount == 0) {
        endNodeAction(currentNode);
    } else {
        endToRootAction(currentNode);
    }
    while (currentNode != startInternode) {
        auto parent = scene->GetParent(currentNode);
        endToRootAction(parent);
        if (!InternodeCheck(scene, currentNode)) {
            endNodeAction(parent);
        }
        currentNode = parent;
    }
}

void IPlantBehaviour::BranchGraphWalker(const std::shared_ptr<Scene> &scene, const Entity &startBranch,
                                        const std::function<void(Entity, Entity)> &rootToEndAction,
                                        const std::function<void(Entity)> &endToRootAction,
                                        const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startBranch;
    while (scene->GetChildrenAmount(currentNode) == 1) {
        Entity child = scene->GetChildren(currentNode)[0];
        if (BranchCheck(scene, child)) {
            rootToEndAction(currentNode, child);
        }
        if (BranchCheck(scene, child)) {
            currentNode = child;
        }
    }
    int trueChildAmount = 0;
    if (scene->GetChildrenAmount(currentNode) != 0) {
        auto children = scene->GetChildren(currentNode);
        for (const auto &child: children) {
            if (BranchCheck(scene, child)) {
                rootToEndAction(currentNode, child);
            }
            if (BranchCheck(scene, child)) {
                BranchGraphWalker(scene, child, rootToEndAction, endToRootAction, endNodeAction);
                if (BranchCheck(scene, child)) {
                    trueChildAmount++;
                }
            }
        }
    }
    if (trueChildAmount == 0) {
        endNodeAction(currentNode);
    } else {
        endToRootAction(currentNode);
    }
    while (currentNode != startBranch) {
        auto parent = scene->GetParent(currentNode);
        endToRootAction(parent);
        if (!BranchCheck(scene, currentNode)) {
            endNodeAction(parent);
        }
        currentNode = parent;
    }
}

void IPlantBehaviour::BranchGraphWalkerRootToEnd(const std::shared_ptr<Scene> &scene, const Entity &startBranch,
                                                 const std::function<void(Entity, Entity)> &rootToEndAction) {
    auto currentNode = startBranch;
    while (scene->GetChildrenAmount(currentNode) == 1) {
        Entity child = scene->GetChildren(currentNode)[0];
        if (BranchCheck(scene, child)) {
            rootToEndAction(currentNode, child);
        }
        if (BranchCheck(scene, child)) {
            currentNode = child;
        }
    }
    if (scene->GetChildrenAmount(currentNode) != 0) {
        auto children = scene->GetChildren(currentNode);
        for (const auto &child: children) {
            if (BranchCheck(scene, child)) {
                rootToEndAction(currentNode, child);
                BranchGraphWalkerRootToEnd(scene, child, rootToEndAction);
            }
        }
    }
}

void IPlantBehaviour::BranchGraphWalkerEndToRoot(const std::shared_ptr<Scene> &scene, const Entity &startBranch,
                                                 const std::function<void(Entity)> &endToRootAction,
                                                 const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startBranch;
    while (scene->GetChildrenAmount(currentNode) == 1) {
        Entity child = scene->GetChildren(currentNode)[0];
        if (BranchCheck(scene, child)) {
            currentNode = child;
        }
    }
    int trueChildAmount = 0;
    if (scene->GetChildrenAmount(currentNode) != 0) {
        auto children = scene->GetChildren(currentNode);
        for (const auto &child: children) {
            if (BranchCheck(scene, child)) {
                BranchGraphWalkerEndToRoot(scene, child, endToRootAction, endNodeAction);
                if (BranchCheck(scene, child)) {
                    trueChildAmount++;
                }
            }
        }
    }
    if (trueChildAmount == 0) {
        endNodeAction(currentNode);
    } else {
        endToRootAction(currentNode);
    }
    while (currentNode != startBranch) {
        auto parent = scene->GetParent(currentNode);
        endToRootAction(parent);
        if (!BranchCheck(scene, currentNode)) {
            endNodeAction(parent);
        }
        currentNode = parent;
    }
}

void IPlantBehaviour::ApplyTropism(const glm::vec3 &targetDir, float tropism, glm::vec3 &front, glm::vec3 &up) {
    const glm::vec3 dir = glm::normalize(targetDir);
    const float dotP = glm::abs(glm::dot(front, dir));
    if (dotP < 0.99f && dotP > -0.99f) {
        const glm::vec3 left = glm::cross(front, dir);
        const float maxAngle = glm::acos(dotP);
        const float rotateAngle = maxAngle * tropism;
        front = glm::normalize(
                glm::rotate(front, glm::min(maxAngle, rotateAngle), left));
        up = glm::normalize(glm::cross(glm::cross(front, up), front));
        // up = glm::normalize(glm::rotate(up, glm::min(maxAngle, rotateAngle),
        // left));
    }
}

std::string IPlantBehaviour::GetTypeName() const {
    return m_typeName;
}

void
IPlantBehaviour::PrepareFoliageMatrices(const std::shared_ptr<Scene> &scene, const MeshGeneratorSettings &settings) {
    scene->ForEach<Transform, GlobalTransform, InternodeInfo>
            (Jobs::Workers(),
             m_internodesQuery,
             [&](int index, Entity entity, Transform &transform, GlobalTransform &globalTransform,
                 InternodeInfo &internodeInfo) {
                 auto internode =
                         scene->GetOrSetPrivateComponent<Internode>(entity).lock();
                 internode->m_foliageMatrices.clear();
                 auto rootGlobalTransform = globalTransform;
                 auto rootEntity = scene->GetRoot(internode->GetOwner());
                 if (rootEntity != entity) {
                     rootGlobalTransform = scene->GetDataComponent<GlobalTransform>(rootEntity);
                 }
                 auto root = scene->GetOrSetPrivateComponent<InternodePlant>(rootEntity).lock();
                 auto inverseGlobalTransform = glm::inverse(rootGlobalTransform.m_value);
                 GlobalTransform relativeGlobalTransform;
                 GlobalTransform relativeParentGlobalTransform;
                 relativeGlobalTransform.m_value = inverseGlobalTransform * globalTransform.m_value;
                 relativeParentGlobalTransform.m_value =
                         inverseGlobalTransform * (glm::inverse(transform.m_value) * globalTransform.m_value);
                 auto plantDescriptor = root->m_plantDescriptor.Get<IPlantDescriptor>();
                 if (!plantDescriptor) return;
                 auto foliagePhyllotaxis = plantDescriptor->m_foliagePhyllotaxis.Get<IInternodeFoliage>();
                 if (foliagePhyllotaxis)
                     foliagePhyllotaxis->GenerateFoliage(internode, internodeInfo,
                                                         relativeGlobalTransform, relativeParentGlobalTransform);
             });
}

Entity IPlantBehaviour::CreateSubtree(const std::shared_ptr<Scene> &scene, const Entity &internodeEntity,
                                      const SubtreeSettings &subtreeSettings) {
    auto subtree = scene->CreateEntity("Subtree");
    if (subtreeSettings.m_enableBase) {
        MeshGeneratorSettings settings;
        settings.m_resolution = subtreeSettings.m_resolution;
        settings.m_subdivision = subtreeSettings.m_subdivision;
        settings.m_vertexColorOnly = true;
        settings.m_enableFoliage = false;
        std::vector<Entity> subtreeInternodes;
        InternodeCollector(scene, internodeEntity, subtreeInternodes, false, subtreeSettings.m_layer);

        if (!subtreeSettings.m_enableBaseInternode) {
            subtreeInternodes.erase(subtreeInternodes.begin());
        }

        PrepareBranchRings(scene, settings);
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        BranchMeshGenerator(scene, subtreeInternodes, vertices, indices, settings);

        auto base = scene->CreateEntity("Base");
        scene->SetParent(base, subtree);
        auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(base).lock();
        auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
        mesh->SetVertices(17, vertices, indices);
        meshRenderer->m_mesh = mesh;
        auto material = ProjectManager::CreateTemporaryAsset<Material>();
        material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        material->m_vertexColorOnly = true;
        meshRenderer->m_material = material;
    }

    if (subtreeSettings.m_enableLines) {
        MeshGeneratorSettings settings;
        settings.m_resolution = subtreeSettings.m_resolution;
        settings.m_subdivision = subtreeSettings.m_subdivision;
        settings.m_vertexColorOnly = true;
        settings.m_enableFoliage = false;
        settings.m_smoothness = subtreeSettings.m_lineSmoothness;
        settings.m_overrideRadius = true;
        settings.m_radius = subtreeSettings.m_lineRadius;
        settings.m_internodeLengthFactor = subtreeSettings.m_lineLengthFactor;
        settings.m_overrideVertexColor = true;
        settings.m_branchVertexColor = glm::vec3(subtreeSettings.m_lineColor);
        std::vector<Entity> subtreeInternodes;
        InternodeCollector(scene, internodeEntity, subtreeInternodes, true, subtreeSettings.m_layer + 1);
        PrepareBranchRings(scene, settings);
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        BranchMeshGenerator(scene, subtreeInternodes, vertices, indices, settings);

        auto lines = scene->CreateEntity("Lines");
        scene->SetParent(lines, subtree);

        auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(lines).lock();
        auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
        mesh->SetVertices(17, vertices, indices);
        meshRenderer->m_mesh = mesh;
        auto material = ProjectManager::CreateTemporaryAsset<Material>();
        material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        material->m_vertexColorOnly = true;
        meshRenderer->m_material = material;
    }

    if (subtreeSettings.m_enablePoints) {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        std::vector<Entity> subtreeInternodes;
        InternodeCollector(scene, internodeEntity, subtreeInternodes, true, subtreeSettings.m_layer);
        auto balls = scene->CreateEntity("Points");
        scene->SetParent(balls, subtree);

        auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(balls).lock();
        auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();

        std::vector<glm::mat4> pointMatrices;
        for (const auto &entity: subtreeInternodes) {
            auto internodeInfo = scene->GetDataComponent<InternodeInfo>(entity);
            GlobalTransform globalTransform = scene->GetDataComponent<GlobalTransform>(entity);
            globalTransform.SetPosition(globalTransform.GetPosition() +
                                        internodeInfo.m_length * (globalTransform.GetRotation() * glm::vec3(0, 0, -1)));
            globalTransform.SetScale(glm::vec3(internodeInfo.m_thickness));
            pointMatrices.emplace_back(globalTransform.m_value);
        }
        auto sphereMesh = DefaultResources::Primitives::Sphere;
        auto &sphereTriangles = sphereMesh->UnsafeGetTriangles();
        auto sphereVerticesSize = sphereMesh->GetVerticesAmount();
        int offset = 0;
        for (const auto &matrix: pointMatrices) {
            Vertex archetype;
            for (auto i = 0; i < sphereMesh->GetVerticesAmount(); i++) {
                archetype.m_position =
                        matrix * glm::vec4(sphereMesh->UnsafeGetVertices()[i].m_position, 1.0f);
                archetype.m_normal = glm::normalize(glm::vec3(
                        matrix * glm::vec4(sphereMesh->UnsafeGetVertices()[i].m_normal, 0.0f)));
                archetype.m_tangent = glm::normalize(glm::vec3(
                        matrix *
                        glm::vec4(sphereMesh->UnsafeGetVertices()[i].m_tangent, 0.0f)));
                archetype.m_texCoord =
                        sphereMesh->UnsafeGetVertices()[i].m_texCoord;
                vertices.push_back(archetype);
            }
            for (auto triangle: sphereTriangles) {
                triangle.x += offset;
                triangle.y += offset;
                triangle.z += offset;
                indices.push_back(triangle.x);
                indices.push_back(triangle.y);
                indices.push_back(triangle.z);
            }
            offset += sphereVerticesSize;
        }

        mesh->SetVertices(17, vertices, indices);
        meshRenderer->m_mesh = mesh;
        auto material = ProjectManager::CreateTemporaryAsset<Material>();
        material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        material->m_vertexColorOnly = false;
        material->m_materialProperties.m_albedoColor = subtreeSettings.m_pointColor;
        meshRenderer->m_material = material;
    }

    if (subtreeSettings.m_enableArrows) {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        std::vector<Entity> subtreeInternodes;
        InternodeCollector(scene, internodeEntity, subtreeInternodes, true, subtreeSettings.m_layer + 1);
        auto balls = scene->CreateEntity("Arrows");
        scene->SetParent(balls, subtree);

        auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(balls).lock();
        auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();

        std::vector<glm::mat4> pointMatrices;
        for (const auto &entity: subtreeInternodes) {
            auto internodeInfo = scene->GetDataComponent<InternodeInfo>(entity);
            GlobalTransform globalTransform = scene->GetDataComponent<GlobalTransform>(entity);
            glm::vec3 front = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
            glm::vec3 up = globalTransform.GetRotation() * glm::vec3(0, 1, 0);
            globalTransform.SetPosition(globalTransform.GetPosition() +
                                        internodeInfo.m_length * subtreeSettings.m_lineLengthFactor * front);
            globalTransform.SetScale(glm::vec3(subtreeSettings.m_lineRadius * 2.0f));
            globalTransform.SetRotation(glm::quatLookAt(up, front));
            pointMatrices.emplace_back(globalTransform.m_value);
        }
        auto sphereMesh = DefaultResources::Primitives::Cone;
        auto &sphereTriangles = sphereMesh->UnsafeGetTriangles();
        auto sphereVerticesSize = sphereMesh->GetVerticesAmount();
        int offset = 0;
        for (const auto &matrix: pointMatrices) {
            Vertex archetype;
            for (auto i = 0; i < sphereMesh->GetVerticesAmount(); i++) {
                archetype.m_position =
                        matrix * glm::vec4(sphereMesh->UnsafeGetVertices()[i].m_position, 1.0f);
                archetype.m_normal = glm::normalize(glm::vec3(
                        matrix * glm::vec4(sphereMesh->UnsafeGetVertices()[i].m_normal, 0.0f)));
                archetype.m_tangent = glm::normalize(glm::vec3(
                        matrix *
                        glm::vec4(sphereMesh->UnsafeGetVertices()[i].m_tangent, 0.0f)));
                archetype.m_texCoord =
                        sphereMesh->UnsafeGetVertices()[i].m_texCoord;
                vertices.push_back(archetype);
            }
            for (auto triangle: sphereTriangles) {
                triangle.x += offset;
                triangle.y += offset;
                triangle.z += offset;
                indices.push_back(triangle.x);
                indices.push_back(triangle.y);
                indices.push_back(triangle.z);
            }
            offset += sphereVerticesSize;
        }

        mesh->SetVertices(17, vertices, indices);
        meshRenderer->m_mesh = mesh;
        auto material = ProjectManager::CreateTemporaryAsset<Material>();
        material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        material->m_vertexColorOnly = false;
        material->m_materialProperties.m_albedoColor = subtreeSettings.m_arrowColor;
        meshRenderer->m_material = material;
    }
    return subtree;
}

void IPlantBehaviour::PrepareBranchRings(const std::shared_ptr<Scene> &scene, const MeshGeneratorSettings &settings) {
    scene->ForEach<GlobalTransform, Transform,
            InternodeInfo>(Jobs::Workers(),
                           m_internodesQuery,
                           [&](int i, Entity entity, GlobalTransform &globalTransform,
                               Transform &transform, InternodeInfo &internodeInfo) {
                               auto internode =
                                       scene->GetOrSetPrivateComponent<Internode>(entity).lock();
                               internode->m_normalDir = globalTransform.GetRotation() * glm::vec3(1, 0, 0);
                               internode->m_rings.clear();
                               auto rootGlobalTransform = globalTransform;
                               auto root = scene->GetRoot(internode->GetOwner());
                               if (root != entity) {
                                   rootGlobalTransform = scene->GetDataComponent<GlobalTransform>(root);
                               }
                               GlobalTransform relativeGlobalTransform;
                               relativeGlobalTransform.m_value =
                                       glm::inverse(rootGlobalTransform.m_value) * globalTransform.m_value;
                               glm::vec3 directionStart =
                                       relativeGlobalTransform.GetRotation() * glm::vec3(0, 0, -1);
                               glm::vec3 directionEnd = directionStart;
                               glm::vec3 positionStart = relativeGlobalTransform.GetPosition();
                               glm::vec3 positionEnd = positionStart +
                                                       internodeInfo.m_length * settings.m_internodeLengthFactor *
                                                       directionStart;
                               float thicknessStart = internodeInfo.m_thickness;
                               float thicknessEnd = internodeInfo.m_thickness;
                               if (root != entity) {
                                   auto parent = scene->GetParent(entity);
                                   if (parent.GetIndex() != 0) {
                                       if (scene->HasDataComponent<InternodeInfo>(parent)) {
                                           auto parentInternodeInfo = scene->GetDataComponent<InternodeInfo>(
                                                   parent);
                                           auto parentGlobalTransform = scene->GetDataComponent<GlobalTransform>(
                                                   parent);
                                           thicknessStart = parentInternodeInfo.m_thickness;
                                           GlobalTransform parentRelativeGlobalTransform;
                                           parentRelativeGlobalTransform.m_value =
                                                   glm::inverse(rootGlobalTransform.m_value) *
                                                   parentGlobalTransform.m_value;
                                           directionStart =
                                                   parentRelativeGlobalTransform.GetRotation() *
                                                   glm::vec3(0, 0, -1);
                                       }
                                   }
                               }
                               if (settings.m_overrideRadius) {
                                   thicknessStart = settings.m_radius;
                                   thicknessEnd = settings.m_radius;
                               }
#pragma region Subdivision internode here.
                               int step = thicknessStart / settings.m_resolution;
                               if (step < 4)
                                   step = 4;
                               if (step % 2 != 0)
                                   step++;
                               internode->m_step = step;
                               int amount = static_cast<int>(0.5f +
                                                             internodeInfo.m_length * settings.m_subdivision);
                               if (amount % 2 != 0)
                                   amount++;
                               BezierCurve curve = BezierCurve(
                                       positionStart,
                                       positionStart +
                                       (settings.m_smoothness ? internodeInfo.m_length / 3.0f : 0.0f) * directionStart,
                                       positionEnd -
                                       (settings.m_smoothness ? internodeInfo.m_length / 3.0f : 0.0f) * directionEnd,
                                       positionEnd);
                               float posStep = 1.0f / static_cast<float>(amount);
                               glm::vec3 dirStep = (directionEnd - directionStart) / static_cast<float>(amount);
                               float radiusStep = (thicknessEnd - thicknessStart) /
                                                  static_cast<float>(amount);

                               for (int i = 1; i < amount; i++) {
                                   float startThickness = static_cast<float>(i - 1) * radiusStep;
                                   float endThickness = static_cast<float>(i) * radiusStep;
                                   if (settings.m_smoothness) {
                                       internode->m_rings.emplace_back(
                                               curve.GetPoint(posStep * (i - 1)), curve.GetPoint(posStep * i),
                                               directionStart + static_cast<float>(i - 1) * dirStep,
                                               directionStart + static_cast<float>(i) * dirStep,
                                               thicknessStart + startThickness, thicknessStart + endThickness);
                                   } else {
                                       internode->m_rings.emplace_back(
                                               curve.GetPoint(posStep * (i - 1)), curve.GetPoint(posStep * i),
                                               directionEnd,
                                               directionEnd,
                                               thicknessStart + startThickness, thicknessStart + endThickness);
                                   }
                               }
                               if (amount > 1)
                                   internode->m_rings.emplace_back(
                                           curve.GetPoint(1.0f - posStep), positionEnd, directionEnd - dirStep,
                                           directionEnd,
                                           thicknessEnd - radiusStep,
                                           thicknessEnd);
                               else
                                   internode->m_rings.emplace_back(positionStart, positionEnd,
                                                                   directionStart, directionEnd, thicknessStart,
                                                                   thicknessEnd);
#pragma endregion

                           }
    );
}

void IPlantBehaviour::BranchMeshGenerator(const std::shared_ptr<Scene> &scene, std::vector<Entity> &internodeEntitites,
                                          std::vector<Vertex> &vertices,
                                          std::vector<unsigned int> &indices,
                                          const MeshGeneratorSettings &settings) {
    int parentStep = -1;
    for (int internodeIndex = 0; internodeIndex < internodeEntitites.size();
         internodeIndex++) {
        auto &internodeEntity = internodeEntitites[internodeIndex];
        auto parent = scene->GetParent(internodeEntity);
        auto internodeGlobalTransform = scene->GetDataComponent<GlobalTransform>(internodeEntity);
        glm::vec3 newNormalDir;
        if (internodeIndex != 0) {
            newNormalDir = scene->GetOrSetPrivateComponent<Internode>(parent).lock()->m_normalDir;
        } else {
            newNormalDir = internodeGlobalTransform.GetRotation() *
                           glm::vec3(1.0f, 0.0f, 0.0f);
        }
        const glm::vec3 front =
                internodeGlobalTransform.GetRotation() *
                glm::vec3(0.0f, 0.0f, -1.0f);
        newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);
        auto internode = scene->GetOrSetPrivateComponent<Internode>(internodeEntity).lock();
        auto branchColor = scene->GetDataComponent<InternodeColor>(internodeEntity);
        internode->m_normalDir = newNormalDir;
        if (internode->m_rings.empty()) {
            continue;
        }
        auto step = internode->m_step;
        // For stitching
        const int pStep = parentStep > 0 ? parentStep : step;
        parentStep = step;

        float angleStep = 360.0f / static_cast<float>(pStep);
        int vertexIndex = vertices.size();
        Vertex archetype;
        if (settings.m_overrideVertexColor) archetype.m_color = settings.m_branchVertexColor;
        else archetype.m_color = branchColor.m_value;

        float textureXStep = 1.0f / pStep * 4.0f;

        const auto startPosition = internode->m_rings.at(0).m_startPosition;
        const auto endPosition = internode->m_rings.back().m_endPosition;
        for (int i = 0; i < pStep; i++) {
            archetype.m_position =
                    internode->m_rings.at(0).GetPoint(newNormalDir, angleStep * i, true);
            float distanceToStart = 0;
            float distanceToEnd = 1;
            const float x =
                    i < pStep / 2 ? i * textureXStep : (pStep - i) * textureXStep;
            archetype.m_texCoord = glm::vec2(x, 0.0f);
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
        int ringSize = internode->m_rings.size();
        for (auto ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            for (auto i = 0; i < step; i++) {
                archetype.m_position = internode->m_rings.at(ringIndex).GetPoint(
                        newNormalDir, angleStep * i, false);
                float distanceToStart = glm::distance(
                        internode->m_rings.at(ringIndex).m_endPosition, startPosition);
                float distanceToEnd = glm::distance(
                        internode->m_rings.at(ringIndex).m_endPosition, endPosition);
                const auto x =
                        i < (step / 2) ? i * textureXStep : (step - i) * textureXStep;
                const auto y = ringIndex % 2 == 0 ? 1.0f : 0.0f;
                archetype.m_texCoord = glm::vec2(x, y);
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

void IPlantBehaviour::InternodeCollector(const std::shared_ptr<Scene> &scene, const Entity &target,
                                         std::vector<Entity> &results, bool onlyCollectEnd, int remainingLayer) {
    if (remainingLayer == 0) return;
    if (scene->IsEntityValid(target) && scene->HasDataComponent<InternodeInfo>(target) &&
        scene->HasPrivateComponent<Internode>(target)) {
        if (!onlyCollectEnd || remainingLayer == 1) results.push_back(target);
        scene->ForEachChild(target, [&](Entity child) {
            InternodeCollector(scene, child, results, onlyCollectEnd, remainingLayer - 1);
        });
    }
}

void IPlantBehaviour::OnInspect() {
    OnMenu();

}
