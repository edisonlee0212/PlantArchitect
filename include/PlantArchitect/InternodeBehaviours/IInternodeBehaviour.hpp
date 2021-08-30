#pragma once

#include <plant_architect_export.h>
#include <Internode.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    struct SpaceColonizationParameters;
    class PLANT_ARCHITECT_API IInternodeBehaviour : public IAsset {
    protected:
#pragma region InternodeFactory
        /**
         * The EntityQuery for filtering all target internodes, must be set when created.
         */
        EntityQuery m_internodesQuery;
        /**
         * The EntityArchetype for creating internodes, must be set when created.
         */
        EntityArchetype m_internodeArchetype;
        EntityRef m_recycleStorageEntity;
        std::mutex m_internodeFactoryLock;
        std::vector<Entity> m_recycledInternodes;

        template<typename T>
        Entity Retrieve(const Entity &parent);

        template<typename T>
        Entity Retrieve();

        void Recycle(const Entity &internode);

        void RecycleSingle(const Entity &internode);

#pragma endregion
#pragma region Helpers
        void TreeNodeWalker(std::vector<Entity> &boundEntities,
                                        std::vector<int> &parentIndices,
                                        const int &parentIndex, const Entity &node);
        void TreeSkinnedMeshGenerator(std::vector<Entity> &internodes,
                                      std::vector<int> &parentIndices,
                                      std::vector<SkinnedVertex> &vertices,
                                      std::vector<unsigned> &indices);
#pragma endregion
    public:
        /**
         * What to do before the growth, and before the resource collection. Mesh, graph calculation...
         */
        virtual void PreProcess() {};

        /**
         * Handle main growth here.
         */
        virtual void Grow() {};

        /**
         * What to do after the growth. Mesh, graph calculation...
         */
        virtual void PostProcess() {};

        /**
         * Generate branch skinned mesh for internodes.
         * @param entities
         */
        virtual void GenerateBranchSkinnedMeshes(const EntityQuery& internodeQuery);

        template<typename T>
        void CreateInternodeMenu(const std::string& menuTitle,
                const std::string &parameterExtension,
                                 const std::function<void(T &params)> &parameterInspector,
                                 const std::function<void(T &params,
                                                          const std::filesystem::path &path)> &parameterDeserializer,
                                 const std::function<void(const T &params,
                                                          const std::filesystem::path &path)> &parameterSerializer,
                                 const std::function<Entity(const T &params,
                                                            const Transform &transform)> &internodeCreator
        );

    };

    template<typename T>
    void IInternodeBehaviour::CreateInternodeMenu(const std::string& menuTitle,
                                                  const std::string &parameterExtension,
                                                  const std::function<void(T &params)> &parameterInspector,
                                                  const std::function<void(T &params,
                                                                           const std::filesystem::path &path)> &parameterDeserializer,
                                                  const std::function<void(const T &params,
                                                                           const std::filesystem::path &path)> &parameterSerializer,
                                                  const std::function<Entity(const T &params,
                                                                             const Transform &transform)> &internodeCreator
    ) {
        if (ImGui::Button("Create...")) {
            ImGui::OpenPopup(menuTitle.c_str());
        }
        const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal(menuTitle.c_str(), nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            static std::vector<T> newPlantParameters;
            static std::vector<glm::vec3> newTreePositions;
            static std::vector<glm::vec3> newTreeRotations;
            static int newTreeAmount = 1;
            static int currentFocusedNewTreeIndex = 0;
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
            ImGui::BeginChild("ChildL", ImVec2(300, 400), true,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Settings")) {
                    static float distance = 10;
                    static float variance = 4;
                    static float yAxisVar = 180.0f;
                    static float xzAxisVar = 0.0f;
                    static int expand = 1;
                    if (ImGui::BeginMenu("Create multiple plants...")) {
                        ImGui::DragFloat("Avg. Y axis rotation", &yAxisVar, 0.01f, 0.0f,
                                         180.0f);
                        ImGui::DragFloat("Avg. XZ axis rotation", &xzAxisVar, 0.01f, 0.0f,
                                         90.0f);
                        ImGui::DragFloat("Avg. Distance", &distance, 0.01f);
                        ImGui::DragFloat("Position variance", &variance, 0.01f);
                        ImGui::DragInt("Expand", &expand, 1, 0, 3);
                        if (ImGui::Button("Apply")) {
                            newTreeAmount = (2 * expand + 1) * (2 * expand + 1);
                            newTreePositions.resize(newTreeAmount);
                            newTreeRotations.resize(newTreeAmount);
                            const auto currentSize = newPlantParameters.size();
                            newPlantParameters.resize(newTreeAmount);
                            for (auto i = currentSize; i < newTreeAmount; i++) {
                                newPlantParameters[i] = newPlantParameters[0];
                            }
                            int index = 0;
                            for (int i = -expand; i <= expand; i++) {
                                for (int j = -expand; j <= expand; j++) {
                                    glm::vec3 value = glm::vec3(i * distance, 0, j * distance);
                                    value.x += glm::linearRand(-variance, variance);
                                    value.z += glm::linearRand(-variance, variance);
                                    newTreePositions[index] = value;
                                    value = glm::vec3(glm::linearRand(-xzAxisVar, xzAxisVar),
                                                      glm::linearRand(-yAxisVar, yAxisVar),
                                                      glm::linearRand(-xzAxisVar, xzAxisVar));
                                    newTreeRotations[index] = value;
                                    index++;
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }
                    ImGui::InputInt("Amount", &newTreeAmount);
                    if (newTreeAmount < 1)
                        newTreeAmount = 1;
                    FileUtils::OpenFile("Import parameters for all", "Parameters", {parameterExtension},
                                        [&](const std::filesystem::path &path) {
                                            parameterDeserializer(newPlantParameters[0], path);
                                            for (int i = 1; i < newPlantParameters.size(); i++)
                                                newPlantParameters[i] = newPlantParameters[0];
                                        });
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
            ImGui::Columns(1);
            if (newTreePositions.size() < newTreeAmount) {
                if (newPlantParameters.empty()) {
                    newPlantParameters.resize(1);
                    newPlantParameters[0] = T();
                }
                const auto currentSize = newTreePositions.size();
                newPlantParameters.resize(newTreeAmount);
                for (auto i = currentSize; i < newTreeAmount; i++) {
                    newPlantParameters[i] = newPlantParameters[0];
                }
                newTreePositions.resize(newTreeAmount);
                newTreeRotations.resize(newTreeAmount);
            }
            for (auto i = 0; i < newTreeAmount; i++) {
                std::string title = "New Tree No.";
                title += std::to_string(i);
                const bool opened = ImGui::TreeNodeEx(
                        title.c_str(), ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                       ImGuiTreeNodeFlags_OpenOnArrow |
                                       ImGuiTreeNodeFlags_NoAutoOpenOnLog |
                                       (currentFocusedNewTreeIndex == i
                                        ? ImGuiTreeNodeFlags_Framed
                                        : ImGuiTreeNodeFlags_FramePadding));
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    currentFocusedNewTreeIndex = i;
                }
                if (opened) {
                    ImGui::TreePush();
                    ImGui::InputFloat3(("Position##" + std::to_string(i)).c_str(),
                                       &newTreePositions[i].x);
                    ImGui::TreePop();
                }
            }

            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
            ImGui::BeginChild("ChildR", ImVec2(400, 400), true,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Parameters")) {
                    FileUtils::OpenFile(
                            "Import parameters", "Parameters", {parameterExtension},
                            [&](const std::filesystem::path &path) {
                                parameterDeserializer(newPlantParameters[currentFocusedNewTreeIndex], path);
                            });

                    FileUtils::SaveFile(
                            "Export parameters", "Parameters", {parameterExtension},
                            [&](const std::filesystem::path &path) {
                                parameterSerializer(newPlantParameters[currentFocusedNewTreeIndex], path);
                            });
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
            ImGui::Columns(1);
            ImGui::PushItemWidth(200);
            parameterInspector(newPlantParameters[currentFocusedNewTreeIndex]);
            ImGui::PopItemWidth();
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::Separator();
            if (ImGui::Button("OK", ImVec2(120, 0))) {
                // Create tree here.
                for (auto i = 0; i < newTreeAmount; i++) {
                    Transform treeTransform;
                    treeTransform.SetPosition(newTreePositions[i]);
                    treeTransform.SetEulerRotation(glm::radians(newTreeRotations[i]));
                    Entity tree = internodeCreator(newPlantParameters[i], treeTransform);
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SetItemDefaultFocus();
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

    }

    template<typename T>
    Entity IInternodeBehaviour::Retrieve(const Entity &parent) {
        Entity retVal;
        std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
        if (!m_recycledInternodes.empty()) {
            retVal = m_recycledInternodes.back();
            retVal.SetParent(parent);
            m_recycledInternodes.pop_back();
            retVal.SetEnabled(true);
            retVal.GetOrSetPrivateComponent<Internode>().lock()->OnRetrieve();
        } else {
            retVal = EntityManager::CreateEntity(m_internodeArchetype, "Internode");
            retVal.SetParent(parent);
            auto internode = retVal.GetOrSetPrivateComponent<Internode>().lock();
            internode->m_resource = std::make_unique<T>();
            internode->OnRetrieve();
        }
        return retVal;
    }

    template<typename T>
    Entity IInternodeBehaviour::Retrieve() {
        Entity retVal;
        std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
        if (!m_recycledInternodes.empty()) {
            retVal = m_recycledInternodes.back();
            m_recycleStorageEntity.Get().RemoveChild(retVal);
            m_recycledInternodes.pop_back();
            retVal.SetEnabled(true);
            retVal.GetOrSetPrivateComponent<Internode>().lock()->OnRetrieve();
        } else {
            retVal = EntityManager::CreateEntity(m_internodeArchetype, "Internode");
            auto internode = retVal.GetOrSetPrivateComponent<Internode>().lock();
            internode->m_resource = std::make_unique<T>();
            internode->OnRetrieve();
        }
        return retVal;
    }


}