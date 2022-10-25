//
// Created by lllll on 10/24/2022.
//

#include "Tree.hpp"
#include "Graphics.hpp"
#include "EditorLayer.hpp"
#include "Application.hpp"

using namespace Orchards;

void Tree::OnInspect() {
    static bool debugVisualization;
    if(ImGui::Button("Grow")){
        m_model.Grow({999});
    }
    ImGui::Checkbox("Visualization", &debugVisualization);
    if (debugVisualization) {
        int version = -1;
        static std::vector<InternodeHandle> sortedInternodeList;
        static std::vector<BranchHandle> sortedBranchList;
        static std::vector<glm::vec4> randomColors;
        if (randomColors.empty()) {
            for (int i = 0; i < 1000; i++) {
                randomColors.push_back(glm::vec4(glm::ballRand(1.0f), 1.0f));
            }
        }
        static std::vector<glm::mat4> matrices;
        static std::vector<glm::vec4> colors;
        static Handle handle;
        bool needUpdate = handle == GetHandle();
        if(ImGui::Button("Update")) needUpdate = true;
        if (needUpdate || m_model.m_targetPlant->GetVersion() != version) {
            version = m_model.m_targetPlant->GetVersion();
            needUpdate = true;
            sortedBranchList = m_model.m_targetPlant->GetSortedBranchList();
            sortedInternodeList = m_model.m_targetPlant->GetSortedInternodeList();
        }
        if (needUpdate) {
            matrices.resize(sortedInternodeList.size());
            colors.resize(sortedInternodeList.size());
            std::vector<std::shared_future<void>> results;
            auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
            Jobs::ParallelFor(sortedInternodeList.size(), [&](unsigned i) {
                auto &internode = m_model.m_targetPlant->RefInternode(sortedInternodeList[i]);
                auto rotation = globalTransform.GetRotation() * internode.m_globalRotation;
                glm::vec3 translation = (globalTransform.m_value * glm::translate(internode.m_globalPosition))[3];
                const auto direction = glm::normalize(rotation * glm::vec3(0, 0, -1));
                const glm::vec3 position2 =
                        translation + internode.m_length * direction;
                rotation = glm::quatLookAt(
                        direction, glm::vec3(direction.y, direction.z, direction.x));
                rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
                const glm::mat4 rotationTransform = glm::mat4_cast(rotation);
                matrices[i] =
                        glm::translate((translation + position2) / 2.0f) *
                        rotationTransform *
                        glm::scale(glm::vec3(
                                internode.m_thickness,
                                glm::distance(translation, position2) / 2.0f,
                                internode.m_thickness));
                colors[i] = randomColors[internode.m_branchHandle];
            }, results);
            for (auto &i: results) i.wait();
        }

        if (!matrices.empty()) {
            auto editorLayer = Application::GetLayer<EditorLayer>();
            Gizmos::DrawGizmoMeshInstancedColored(
                    DefaultResources::Primitives::Cylinder, editorLayer->m_sceneCamera,
                    editorLayer->m_sceneCameraPosition,
                    editorLayer->m_sceneCameraRotation,
                    *reinterpret_cast<std::vector<glm::vec4> *>(&colors),
                    *reinterpret_cast<std::vector<glm::mat4> *>(&matrices),
                    glm::mat4(1.0f), 1.0f);
        }
    }
}

void Tree::OnCreate() {
    m_model.Initialize();
}

void Tree::OnDestroy() {
    m_model.Clear();
}
