//
// Created by lllll on 8/27/2021.
//

#include "IPlantBehaviour.hpp"
#include "Internode.hpp"
#include "Curve.hpp"
#include "PlantLayer.hpp"
#include "IInternodePhyllotaxis.hpp"
#include "InternodeFoliage.hpp"
#include "PlantDataComponents.hpp"
#include "TransformLayer.hpp"

using namespace PlantArchitect;

void IPlantBehaviour::UpdateBranches() {
    std::vector<Entity> roots;
    m_rootsQuery.ToEntityArray(Entities::GetCurrentScene(), roots);
    for (const auto &root: roots) {
        if (!RootCheck(root)) return;
        Entity rootInternode, rootBranch;
        root.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (InternodeCheck(child)) rootInternode = child;
            else if (BranchCheck(child)) rootBranch = child;
        });
        if (!rootInternode.IsValid() || !rootBranch.IsValid()) return;
        auto children = rootBranch.GetChildren();
        for (const auto &i: children) Entities::DeleteEntity(Entities::GetCurrentScene(), i);
        auto branch = rootBranch.GetOrSetPrivateComponent<Branch>().lock();
        branch->m_internodeChain.clear();
        BranchGraphWalkerRootToEnd(rootBranch, [](Entity parent, Entity child) {
            child.GetOrSetPrivateComponent<Branch>().lock()->m_internodeChain.clear();
        });
        UpdateBranchHelper(rootBranch, rootInternode);
        {
            auto branchInfo = rootBranch.GetDataComponent<BranchInfo>();
            auto branchStartInternodeGT = branch->m_internodeChain.front().GetDataComponent<GlobalTransform>();
            auto branchEndInternodeGT = branch->m_internodeChain.back().GetDataComponent<GlobalTransform>();
            auto branchStartInternodeInfo = branch->m_internodeChain.front().GetDataComponent<InternodeInfo>();
            auto branchEndInternodeInfo = branch->m_internodeChain.back().GetDataComponent<InternodeInfo>();
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
            if (branchInfo.m_length > 0) branchGT.m_value = glm::translate(branchEndPosition) * glm::mat4_cast(
                        glm::quatLookAt(glm::normalize(branchEndPosition - branchStartPosition),
                                        rotation * glm::vec3(0, 1, 0))) * glm::scale(glm::vec3(1.0f));
            else branchGT.SetPosition(branchEndPosition);
            branchInfo.m_thickness = branchStartInternodeInfo.m_thickness;

            rootBranch.SetDataComponent(branchGT);
            rootBranch.SetDataComponent(branchInfo);
        }
        BranchGraphWalkerRootToEnd(rootBranch, [](Entity parent, Entity child) {
            auto branch = child.GetOrSetPrivateComponent<Branch>().lock();
            auto branchInfo = child.GetDataComponent<BranchInfo>();
            auto branchStartInternodeGT = branch->m_internodeChain.front().GetDataComponent<GlobalTransform>();
            auto branchEndInternodeGT = branch->m_internodeChain.back().GetDataComponent<GlobalTransform>();
            auto branchStartInternodeInfo = branch->m_internodeChain.front().GetDataComponent<InternodeInfo>();
            auto branchEndInternodeInfo = branch->m_internodeChain.back().GetDataComponent<InternodeInfo>();
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
            if (branchInfo.m_length > 0) branchGT.m_value = glm::translate(branchEndPosition) * glm::mat4_cast(
                        glm::quatLookAt(glm::normalize(branchEndPosition - branchStartPosition),
                                        rotation * glm::vec3(0, 1, 0))) * glm::scale(glm::vec3(1.0f));
            else branchGT.SetPosition(branchEndPosition);

            branchInfo.m_thickness = branchStartInternodeInfo.m_thickness;

            child.SetDataComponent(branchGT);
            child.SetDataComponent(branchInfo);
        });
        Application::GetLayer<TransformLayer>()->CalculateTransformGraphForDescendents(
                Entities::GetCurrentScene(),
                rootBranch);
    }
}

void IPlantBehaviour::DestroyInternode(const Entity &internode) {
    std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
    Entities::DeleteEntity(Entities::GetCurrentScene(), internode);
}

void IPlantBehaviour::DestroyBranch(const Entity &branch) {
    std::lock_guard<std::mutex> lockGuard(m_branchFactoryLock);
    auto internode = branch.GetOrSetPrivateComponent<Branch>().lock()->m_currentInternode.Get();
    if (InternodeCheck(internode))DestroyInternode(internode);
    Entities::DeleteEntity(Entities::GetCurrentScene(), branch);
}

void
IPlantBehaviour::GenerateSkinnedMeshes(float subdivision,
                                       float resolution) {
    //UpdateBranches();
    std::vector<Entity> currentRoots;
    m_rootsQuery.ToEntityArray(Entities::GetCurrentScene(), currentRoots);
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
            auto children = currentRoots[plantIndex].GetChildren();
            for (const auto &child: children) {
                if (BranchCheck(child)) {
                    BranchCollector(boundEntitiesLists[plantIndex],
                                    parentIndicesLists[plantIndex], -1, child);
                    break;
                }
            }
        }).share());
    }
    for (const auto &i: results)
        i.wait();

#pragma region Prepare rings for branch mesh.
    Entities::ForEach<GlobalTransform, Transform,
            InternodeInfo>(Entities::GetCurrentScene(),
                           Jobs::Workers(),
                           m_internodesQuery,
                           [resolution, subdivision](int i, Entity entity, GlobalTransform &globalTransform,
                                                     Transform &transform, InternodeInfo &internodeInfo) {
                               auto internode =
                                       entity.GetOrSetPrivateComponent<Internode>().lock();
                               internode->m_rings.clear();
                               auto rootGlobalTransform = globalTransform;
                               auto root = internode->GetOwner().GetRoot();
                               if (root != entity) {
                                   rootGlobalTransform = root.GetDataComponent<GlobalTransform>();
                               }
                               GlobalTransform relativeGlobalTransform;
                               relativeGlobalTransform.m_value =
                                       glm::inverse(rootGlobalTransform.m_value) * globalTransform.m_value;
                               glm::vec3 directionStart = relativeGlobalTransform.GetRotation() * glm::vec3(0, 0, -1);
                               glm::vec3 directionEnd = directionStart;
                               glm::vec3 positionStart = relativeGlobalTransform.GetPosition();
                               glm::vec3 positionEnd = positionStart + internodeInfo.m_length * directionStart;
                               float thicknessStart = internodeInfo.m_thickness;
                               if (root != entity) {
                                   auto parent = entity.GetParent();
                                   if (!parent.IsNull()) {
                                       if (parent.HasDataComponent<InternodeInfo>()) {
                                           auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                                           auto parentGlobalTransform = parent.GetDataComponent<GlobalTransform>();
                                           thicknessStart = parentInternodeInfo.m_thickness;
                                           GlobalTransform parentRelativeGlobalTransform;
                                           parentRelativeGlobalTransform.m_value =
                                                   glm::inverse(rootGlobalTransform.m_value) *
                                                   parentGlobalTransform.m_value;
                                           directionStart =
                                                   parentRelativeGlobalTransform.GetRotation() * glm::vec3(0, 0, -1);
                                       }
                                   }
                               }
#pragma region Subdivision internode here.
                               int step = thicknessStart / resolution;
                               if (step < 4)
                                   step = 4;
                               if (step % 2 != 0)
                                   step++;
                               internode->m_step = step;
                               int amount = static_cast<int>(0.5f + internodeInfo.m_length * subdivision);
                               if (amount % 2 != 0)
                                   amount++;
                               BezierCurve curve = BezierCurve(
                                       positionStart, positionStart + internodeInfo.m_length / 3.0f * directionStart,
                                       positionEnd - internodeInfo.m_length / 3.0f * directionEnd, positionEnd);
                               float posStep = 1.0f / static_cast<float>(amount);
                               glm::vec3 dirStep = (directionEnd - directionStart) / static_cast<float>(amount);
                               float radiusStep = (internodeInfo.m_thickness - thicknessStart) /
                                                  static_cast<float>(amount);

                               for (int i = 1; i < amount; i++) {
                                   float startThickness = static_cast<float>(i - 1) * radiusStep;
                                   float endThickness = static_cast<float>(i) * radiusStep;
                                   internode->m_rings.emplace_back(
                                           curve.GetPoint(posStep * (i - 1)), curve.GetPoint(posStep * i),
                                           directionStart + static_cast<float>(i - 1) * dirStep,
                                           directionStart + static_cast<float>(i) * dirStep,
                                           thicknessStart + startThickness, thicknessStart + endThickness);
                               }
                               if (amount > 1)
                                   internode->m_rings.emplace_back(
                                           curve.GetPoint(1.0f - posStep), positionEnd, directionEnd - dirStep,
                                           directionEnd,
                                           internodeInfo.m_thickness - radiusStep,
                                           internodeInfo.m_thickness);
                               else
                                   internode->m_rings.emplace_back(positionStart, positionEnd,
                                                                   directionStart, directionEnd, thicknessStart,
                                                                   internodeInfo.m_thickness);
#pragma endregion

                           }
    );
#pragma endregion
#pragma region Prepare foliage transforms.
    std::mutex mutex;
    Entities::ForEach<Transform, GlobalTransform, InternodeInfo>
            (Entities::GetCurrentScene(),
             Jobs::Workers(),
             m_internodesQuery,
             [&](int index, Entity entity, Transform &transform, GlobalTransform &globalTransform,
                 InternodeInfo &internodeInfo) {
                 if (entity.GetChildrenAmount() != 0) return;
                 auto internode =
                         entity.GetOrSetPrivateComponent<Internode>().lock();
                 internode->m_foliageMatrices.clear();
                 auto rootGlobalTransform = globalTransform;
                 auto root = internode->GetOwner().GetRoot();
                 if (root != entity) {
                     rootGlobalTransform = root.GetDataComponent<GlobalTransform>();
                 }
                 auto inverseGlobalTransform = glm::inverse(rootGlobalTransform.m_value);
                 GlobalTransform relativeGlobalTransform;
                 GlobalTransform relativeParentGlobalTransform;
                 relativeGlobalTransform.m_value = inverseGlobalTransform * globalTransform.m_value;
                 relativeParentGlobalTransform.m_value =
                         inverseGlobalTransform * (glm::inverse(transform.m_value) * globalTransform.m_value);
                 auto foliage = internode->m_foliage.Get<InternodeFoliage>();
                 if (foliage)
                     foliage->Generate(internode, internodeInfo,
                                       relativeGlobalTransform, relativeParentGlobalTransform);
             });
#pragma endregion
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        const auto &root = currentRoots[plantIndex];
        auto children = root.GetChildren();
        Entity rootInternode, rootBranch;
        for (const auto &child: children) {
            if (InternodeCheck(child)) rootInternode = child;
            else if (BranchCheck(child)) rootBranch = child;
        }
        Entity branchMesh, foliageMesh;
        PrepareInternodeForSkeletalAnimation(root, branchMesh, foliageMesh);
        {
#pragma region Branch mesh
            auto animator = branchMesh.GetOrSetPrivateComponent<Animator>().lock();
            auto skinnedMeshRenderer = branchMesh.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            skinnedMeshRenderer->SetEnabled(true);
            auto internode = rootInternode.GetOrSetPrivateComponent<Internode>().lock();
            const auto plantGlobalTransform =
                    root.GetDataComponent<GlobalTransform>();
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
            animator->Setup(names, offsetMatrices);
            skinnedMeshRenderer->SetRagDoll(true);
            skinnedMeshRenderer->SetRagDollBoundEntities(boundEntitiesLists[plantIndex], false);
#pragma endregion
            std::vector<unsigned> skinnedIndices;
            std::vector<SkinnedVertex> skinnedVertices;
            BranchSkinnedMeshGenerator(boundEntitiesLists[plantIndex],
                                       parentIndicesLists[plantIndex],
                                       skinnedVertices, skinnedIndices);
            if (!skinnedVertices.empty()) {
                auto skinnedMesh = AssetManager::CreateAsset<SkinnedMesh>();
                skinnedMesh->SetVertices(
                        17, skinnedVertices, skinnedIndices);
                skinnedMesh
                        ->m_boneAnimatorIndices = boneIndicesLists[plantIndex];
                skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(skinnedMesh);
            }
#pragma endregion
        }
        {

#pragma region Foliage mesh
            auto animator = foliageMesh.GetOrSetPrivateComponent<Animator>().lock();
            auto skinnedMeshRenderer = foliageMesh.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
            auto material = skinnedMeshRenderer->m_material.Get<Material>();
            skinnedMeshRenderer->SetEnabled(true);
            auto internode = rootInternode.GetOrSetPrivateComponent<Internode>().lock();
            auto foliage = internode->m_foliage.Get<InternodeFoliage>();
            if (foliage && foliage->m_foliageTexture.Get<Texture2D>())
                material->m_albedoTexture = foliage->m_foliageTexture.Get<Texture2D>();
            material->m_albedoColor = foliage->m_foliageColor;
            const auto plantGlobalTransform =
                    root.GetDataComponent<GlobalTransform>();
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
            animator->Setup(names, offsetMatrices);
            skinnedMeshRenderer->SetRagDoll(true);
            skinnedMeshRenderer->SetRagDollBoundEntities(boundEntitiesLists[plantIndex], false);
#pragma endregion
            std::vector<unsigned> skinnedIndices;
            std::vector<SkinnedVertex> skinnedVertices;
            FoliageSkinnedMeshGenerator(boundEntitiesLists[plantIndex],
                                        parentIndicesLists[plantIndex],
                                        skinnedVertices, skinnedIndices);
            if (!skinnedVertices.empty()) {
                auto skinnedMesh = AssetManager::CreateAsset<SkinnedMesh>();
                skinnedMesh->SetVertices(
                        17, skinnedVertices, skinnedIndices);
                skinnedMesh
                        ->m_boneAnimatorIndices = boneIndicesLists[plantIndex];
                skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(skinnedMesh);
            }
#pragma endregion
        }
    }
}

void IPlantBehaviour::BranchCollector(std::vector<Entity> &boundEntities, std::vector<int> &parentIndices,
                                      const int &parentIndex, const Entity &node) {
    if (!BranchCheck(node)) return;
    boundEntities.push_back(node);
    parentIndices.push_back(parentIndex);
    const size_t currentIndex = boundEntities.size() - 1;
    node.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (BranchCheck(child)) BranchCollector(boundEntities, parentIndices, currentIndex, child);
    });

}

void IPlantBehaviour::FoliageSkinnedMeshGenerator(std::vector<Entity> &entities,
                                                  std::vector<int> &parentIndices,
                                                  std::vector<SkinnedVertex> &vertices,
                                                  std::vector<unsigned int> &indices) {
    auto quadMesh = DefaultResources::Primitives::Quad;
    auto &quadTriangles = quadMesh->UnsafeGetTriangles();
    auto quadVerticesSize = quadMesh->GetVerticesAmount();
    size_t offset = 0;
    for (int internodeIndex = 0; internodeIndex < entities.size();
         internodeIndex++) {
        int parentIndex = 0;
        if (internodeIndex != 0) parentIndex = parentIndices[internodeIndex];
        auto &entity = entities[internodeIndex];
        auto internodeGlobalTransform = entity.GetDataComponent<GlobalTransform>();
        auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
        glm::vec3 newNormalDir;
        if (internodeIndex != 0) {
            newNormalDir = entities[parentIndex].GetOrSetPrivateComponent<Internode>().lock()->m_normalDir;
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
            for (auto i = 0; i < quadMesh->GetVerticesAmount(); i++) {
                archetype.m_position =
                        matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_position, 1.0f);
                archetype.m_normal = glm::normalize(glm::vec3(
                        matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_normal, 0.0f)));
                archetype.m_tangent = glm::normalize(glm::vec3(
                        matrix *
                        glm::vec4(quadMesh->UnsafeGetVertices()[i].m_tangent, 0.0f)));
                archetype.m_texCoords =
                        quadMesh->UnsafeGetVertices()[i].m_texCoords;
                archetype.m_bondId = glm::ivec4(internodeIndex, -1, -1, -1);
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

void IPlantBehaviour::BranchSkinnedMeshGenerator(std::vector<Entity> &entities, std::vector<int> &parentIndices,
                                                 std::vector<SkinnedVertex> &vertices,
                                                 std::vector<unsigned int> &indices) {
    int parentStep = -1;
    for (int branchIndex = 0; branchIndex < entities.size();
         branchIndex++) {
        int parentIndex = 0;
        if (branchIndex != 0) parentIndex = parentIndices[branchIndex];
        auto &branchEntity = entities[branchIndex];
        auto branchGlobalTransform = branchEntity.GetDataComponent<GlobalTransform>();
        auto branch = branchEntity.GetOrSetPrivateComponent<Branch>().lock();
        for (const auto &internodeEntity: branch->m_internodeChain) {
            auto internodeGlobalTransform = internodeEntity.GetDataComponent<GlobalTransform>();
            glm::vec3 newNormalDir;
            if (branchIndex != 0) {
                newNormalDir = entities[parentIndex].GetOrSetPrivateComponent<Internode>().lock()->m_normalDir;
            } else {
                newNormalDir = internodeGlobalTransform.GetRotation() *
                               glm::vec3(1.0f, 0.0f, 0.0f);
            }
            const glm::vec3 front =
                    internodeGlobalTransform.GetRotation() *
                    glm::vec3(0.0f, 0.0f, -1.0f);
            newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);
            auto internode = internodeEntity.GetOrSetPrivateComponent<Internode>().lock();
            auto branchColor = internodeEntity.GetDataComponent<InternodeColor>();
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
            SkinnedVertex archetype;
            archetype.m_color = branchColor.m_value;
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
}

void
IPlantBehaviour::PrepareInternodeForSkeletalAnimation(const Entity &entity, Entity &branchMesh, Entity &foliageMesh) {
    entity.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (child.GetName() == "BranchMesh") {
            branchMesh = child;
        } else if (child.GetName() == "FoliageMesh") {
            foliageMesh = child;
        }
    });

    {
        if (branchMesh.IsNull()) branchMesh = Entities::CreateEntity(Entities::GetCurrentScene(), "BranchMesh");
        auto animator = branchMesh.GetOrSetPrivateComponent<Animator>().lock();
        auto skinnedMeshRenderer =
                branchMesh.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
        skinnedMeshRenderer->m_skinnedMesh = AssetManager::CreateAsset<SkinnedMesh>();
        auto skinnedMat = AssetManager::LoadMaterial(
                DefaultResources::GLPrograms::StandardSkinnedProgram);
        skinnedMeshRenderer->m_material = skinnedMat;
        skinnedMat->m_albedoColor = glm::vec3(40.0f / 255, 15.0f / 255, 0.0f);
        skinnedMat->m_roughness = 1.0f;
        skinnedMat->m_metallic = 0.0f;
        skinnedMeshRenderer->m_animator = branchMesh.GetOrSetPrivateComponent<Animator>().lock();
    }
    {
        if (foliageMesh.IsNull()) foliageMesh = Entities::CreateEntity(Entities::GetCurrentScene(), "FoliageMesh");
        auto animator = foliageMesh.GetOrSetPrivateComponent<Animator>().lock();
        auto skinnedMeshRenderer =
                foliageMesh.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
        skinnedMeshRenderer->m_skinnedMesh = AssetManager::CreateAsset<SkinnedMesh>();
        auto skinnedMat = AssetManager::LoadMaterial(
                DefaultResources::GLPrograms::StandardSkinnedProgram);
        skinnedMeshRenderer->m_material = skinnedMat;
        skinnedMat->m_albedoColor = glm::vec3(0.0f, 1.0f, 0.0f);
        skinnedMat->m_roughness = 1.0f;
        skinnedMat->m_metallic = 0.0f;
        skinnedMat->m_cullingMode = MaterialCullingMode::Off;
        skinnedMeshRenderer->m_animator = foliageMesh.GetOrSetPrivateComponent<Animator>().lock();
    }
    branchMesh.SetParent(entity);
    foliageMesh.SetParent(entity);
}

Entity IPlantBehaviour::CreateBranchHelper(const Entity &parent, const Entity &internode) {
    Entity retVal;
    std::lock_guard<std::mutex> lockGuard(m_branchFactoryLock);
    retVal = Entities::CreateEntity(Entities::GetCurrentScene(), m_branchArchetype, "Branch");
    retVal.SetParent(parent);
    BranchInfo branchInfo;
    retVal.SetDataComponent(branchInfo);
    auto parentBranch = parent.GetOrSetPrivateComponent<Branch>().lock();
    auto branch = retVal.GetOrSetPrivateComponent<Branch>().lock();
    branch->m_branchPhysicsParameters = parentBranch->m_branchPhysicsParameters;
    branch->m_currentRoot = parentBranch->m_currentRoot;
    branch->m_currentInternode = internode;
    return retVal;
}

void
IPlantBehaviour::InternodeGraphWalker(const Entity &startInternode,
                                      const std::function<void(Entity, Entity)> &rootToEndAction,
                                      const std::function<void(Entity)> &endToRootAction,
                                      const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startInternode;
    while (currentNode.GetChildrenAmount() == 1) {
        Entity child = currentNode.GetChildren()[0];
        if (InternodeCheck(child)) {
            rootToEndAction(currentNode, child);
        }
        if (InternodeCheck(child)) {
            currentNode = child;
        }
    }
    int trueChildAmount = 0;
    if (currentNode.GetChildrenAmount() != 0) {
        auto children = currentNode.GetChildren();
        for (const auto &child: children) {
            if (InternodeCheck(child)) {
                rootToEndAction(currentNode, child);
            }
            if (InternodeCheck(child)) {
                InternodeGraphWalker(child, rootToEndAction, endToRootAction, endNodeAction);
                if (InternodeCheck(child)) {
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
        auto parent = currentNode.GetParent();
        endToRootAction(parent);
        if (!InternodeCheck(currentNode)) {
            endNodeAction(parent);
        }
        currentNode = parent;
    }
}

bool IPlantBehaviour::InternodeCheck(const Entity &target) {
    return target.IsValid() && target.IsEnabled() && target.HasDataComponent<InternodeInfo>() &&
           target.HasPrivateComponent<Internode>() &&
           InternalInternodeCheck(target);
}

bool IPlantBehaviour::RootCheck(const Entity &target) {
    return target.IsValid() && target.IsEnabled() && target.HasDataComponent<RootInfo>() &&
           target.HasPrivateComponent<Root>() &&
           InternalRootCheck(target);
}

bool IPlantBehaviour::BranchCheck(const Entity &target) {
    return target.IsValid() && target.IsEnabled() && target.HasDataComponent<BranchInfo>() &&
           target.HasPrivateComponent<Branch>() &&
           InternalBranchCheck(target);
}

void IPlantBehaviour::UpdateBranchHelper(const Entity &currentBranch, const Entity &currentInternode) {
    int trueChildAmount = 0;
    currentInternode.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (InternodeCheck(child)) trueChildAmount++;
    });
    currentBranch.GetOrSetPrivateComponent<Branch>().lock()->m_internodeChain.push_back(currentInternode);
    if (trueChildAmount > 1) {
        BranchInfo branchInfo;
        branchInfo.m_endNode = false;
        currentBranch.SetDataComponent(branchInfo);
        currentInternode.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!InternodeCheck(child)) return;
            auto newBranch = CreateBranch(currentBranch, child);
            UpdateBranchHelper(newBranch, child);
        });
    } else if (trueChildAmount == 1) {
        currentInternode.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!InternodeCheck(child)) return;
            UpdateBranchHelper(currentBranch, child);
        });
    }

}

void IPlantBehaviour::InternodeGraphWalkerRootToEnd(const Entity &startInternode,
                                                    const std::function<void(Entity, Entity)> &rootToEndAction) {
    auto currentNode = startInternode;
    while (currentNode.GetChildrenAmount() == 1) {
        Entity child = currentNode.GetChildren()[0];
        if (InternodeCheck(child)) {
            rootToEndAction(currentNode, child);
        }
        if (InternodeCheck(child)) {
            currentNode = child;
        }
    }
    if (currentNode.GetChildrenAmount() != 0) {
        auto children = currentNode.GetChildren();
        for (const auto &child: children) {
            if (InternodeCheck(child)) {
                rootToEndAction(currentNode, child);
                InternodeGraphWalkerRootToEnd(child, rootToEndAction);
            }
        }
    }
}

void IPlantBehaviour::InternodeGraphWalkerEndToRoot(const Entity &startInternode,
                                                    const std::function<void(Entity)> &endToRootAction,
                                                    const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startInternode;
    while (currentNode.GetChildrenAmount() == 1) {
        Entity child = currentNode.GetChildren()[0];
        if (child.IsValid()) {
            currentNode = child;
        }
    }
    int trueChildAmount = 0;
    if (currentNode.GetChildrenAmount() != 0) {
        auto children = currentNode.GetChildren();
        for (const auto &child: children) {
            if (InternodeCheck(child)) {
                InternodeGraphWalkerEndToRoot(child, endToRootAction, endNodeAction);
                if (InternodeCheck(child)) {
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
        auto parent = currentNode.GetParent();
        endToRootAction(parent);
        if (!InternodeCheck(currentNode)) {
            endNodeAction(parent);
        }
        currentNode = parent;
    }
}

void IPlantBehaviour::BranchGraphWalker(const Entity &startBranch,
                                        const std::function<void(Entity, Entity)> &rootToEndAction,
                                        const std::function<void(Entity)> &endToRootAction,
                                        const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startBranch;
    while (currentNode.GetChildrenAmount() == 1) {
        Entity child = currentNode.GetChildren()[0];
        if (BranchCheck(child)) {
            rootToEndAction(currentNode, child);
        }
        if (BranchCheck(child)) {
            currentNode = child;
        }
    }
    int trueChildAmount = 0;
    if (currentNode.GetChildrenAmount() != 0) {
        auto children = currentNode.GetChildren();
        for (const auto &child: children) {
            if (BranchCheck(child)) {
                rootToEndAction(currentNode, child);
            }
            if (BranchCheck(child)) {
                BranchGraphWalker(child, rootToEndAction, endToRootAction, endNodeAction);
                if (BranchCheck(child)) {
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
        auto parent = currentNode.GetParent();
        endToRootAction(parent);
        if (!BranchCheck(currentNode)) {
            endNodeAction(parent);
        }
        currentNode = parent;
    }
}

void IPlantBehaviour::BranchGraphWalkerRootToEnd(const Entity &startBranch,
                                                 const std::function<void(Entity, Entity)> &rootToEndAction) {
    auto currentNode = startBranch;
    while (currentNode.GetChildrenAmount() == 1) {
        Entity child = currentNode.GetChildren()[0];
        if (BranchCheck(child)) {
            rootToEndAction(currentNode, child);
        }
        if (BranchCheck(child)) {
            currentNode = child;
        }
    }
    if (currentNode.GetChildrenAmount() != 0) {
        auto children = currentNode.GetChildren();
        for (const auto &child: children) {
            if (BranchCheck(child)) {
                rootToEndAction(currentNode, child);
                BranchGraphWalkerRootToEnd(child, rootToEndAction);
            }
        }
    }
}

void IPlantBehaviour::BranchGraphWalkerEndToRoot(const Entity &startBranch,
                                                 const std::function<void(Entity)> &endToRootAction,
                                                 const std::function<void(Entity)> &endNodeAction) {
    auto currentNode = startBranch;
    while (currentNode.GetChildrenAmount() == 1) {
        Entity child = currentNode.GetChildren()[0];
        if (child.IsValid()) {
            currentNode = child;
        }
    }
    int trueChildAmount = 0;
    if (currentNode.GetChildrenAmount() != 0) {
        auto children = currentNode.GetChildren();
        for (const auto &child: children) {
            if (BranchCheck(child)) {
                BranchGraphWalkerEndToRoot(child, endToRootAction, endNodeAction);
                if (BranchCheck(child)) {
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
        auto parent = currentNode.GetParent();
        endToRootAction(parent);
        if (!BranchCheck(currentNode)) {
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
