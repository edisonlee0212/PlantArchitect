#pragma once

#include <plant_architect_export.h>
#include "Internode.hpp"
#include "Root.hpp"
#include "Branch.hpp"
#include <Serialization.hpp>
#include "InternodeFoliage.hpp"
#include "InternodeLayer.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    struct SpaceColonizationParameters;
    class PLANT_ARCHITECT_API IPlantBehaviour : public IAsset {
        friend class InternodeLayer;
    protected:
#pragma region InternodeFactory
        /**
         * The EntityQuery for filtering all target internodes, must be set when created.
         */
        EntityQuery m_internodesQuery;
        /**
         * The EntityQuery for filtering all target roots, must be set when created.
         */
        EntityQuery m_rootsQuery;
        /**
         * The EntityQuery for filtering all target branches, must be set when created.
         */
        EntityQuery m_branchesQuery;
        /**
         * The EntityArchetype for creating internodes, must be set when created.
         */
        EntityArchetype m_internodeArchetype;
        /**
         * The EntityArchetype for creating roots, must be set when created.
         */
        EntityArchetype m_rootArchetype;
        /**
         * The EntityArchetype for creating branches, must be set when created.
         */
        EntityArchetype m_branchArchetype;

        std::mutex m_internodeFactoryLock;
        std::mutex m_branchFactoryLock;
        /**
         * Get or create an internode.
         * @tparam T The type of resource that will be added to the internode.
         * @param parent The parent of the internode.
         * @return An entity represent the internode.
         */
        template<typename T>
        Entity CreateHelper(const Entity &parent);

        /**
         * Get or create an branch.
         * @param parent The parent of the branch.
         * @return An entity represent the branch.
         */
        Entity CreateBranchHelper(const Entity &parent);

        /**
         * Get or create a root.
         * @tparam T The type of resource that will be added to the internode.
         * @return An entity represent the root internode.
         */
        template<typename T>
        Entity CreateRootHelper(Entity& rootInternode, Entity& rootBranch);
#pragma endregion
#pragma region Helpers

        void TreeNodeCollector(std::vector<Entity> &boundEntities,
                               std::vector<int> &parentIndices,
                               const int &parentIndex, const Entity &node, const Entity &root);

        void BranchSkinnedMeshGenerator(std::vector<Entity> &entities,
                                        std::vector<int> &parentIndices,
                                        std::vector<SkinnedVertex> &vertices,
                                        std::vector<unsigned> &indices);

        void FoliageSkinnedMeshGenerator(std::vector<Entity> &entities,
                                         std::vector<int> &parentIndices,
                                         std::vector<SkinnedVertex> &vertices,
                                         std::vector<unsigned> &indices);

        void PrepareInternodeForSkeletalAnimation(const Entity &entity, Entity &branchMesh, Entity &foliage);

#pragma endregion

        virtual bool InternalInternodeCheck(const Entity &target) = 0;
        virtual bool InternalRootCheck(const Entity &target) = 0;
        virtual bool InternalBranchCheck(const Entity &target) = 0;
    public:

        void ApplyTropism(const glm::vec3 &targetDir, float tropism, glm::vec3 &front, glm::vec3 &up);

        virtual Entity CreateRoot(Entity& rootInternode, Entity& rootBranch) = 0;
        virtual Entity CreateBranch(const Entity &parent) = 0;
        virtual Entity CreateInternode(const Entity &parent) = 0;

        /*
         * Disable and recycle the internode and all its descendents.
         */
        void DestroyInternode(const Entity &internode);

        /**
         * Handle main growth here.
         */
        virtual void Grow(int iteration) {};

        /**
         * Generate skinned mesh for internodes.
         * @param entities
         */
        void
        GenerateSkinnedMeshes(float subdivision = 4.0f, float resolution = 0.02f);


        /**
         * A helper method that traverse target plant from root internode to end internodes, and come back from end to root.
         * @param node Walker node, should be same as root. It's your responsibility to make sure the root is valid.
         * @param rootToEndAction The action to take during traversal from root to end. You must not delete the parent during the action.
         * @param endToRootAction The action to take during traversal from end to root. You must not delete the parent during the action.
         * @param endNodeAction The action to take for end nodes. You must not delete the end node during the action.
         */
        void TreeGraphWalker(const Entity &startInternode,
                             const std::function<void(Entity parent, Entity child)> &rootToEndAction,
                             const std::function<void(Entity parent)> &endToRootAction,
                             const std::function<void(Entity endNode)> &endNodeAction);

        /**
         * A helper method that traverse target plant from root internode to end internodes.
         * @param node Walker node, should be same as root. It's your responsibility to make sure the root is valid.
         * @param rootToEndAction The action to take during traversal from root to end.
         */
        void TreeGraphWalkerRootToEnd(const Entity &startInternode,
                                      const std::function<void(Entity parent, Entity child)> &rootToEndAction);

        /**
         * A helper method that traverse target plant from end internodes to root internode.
         * @param node Walker node, should be same as root. It's your responsibility to make sure the root is valid.
         * @param endToRootAction The action to take during traversal from end to root. You must not delete the parent during the action.
         * @param endNodeAction The action to take for end nodes. You must not delete the end node during the action.
         */
        void TreeGraphWalkerEndToRoot(const Entity &startInternode,
                                      const std::function<void(Entity parent)> &endToRootAction,
                                      const std::function<void(Entity endNode)> &endNodeAction);


        /**
         * Check if the entity is valid internode.
         * @param target Target for check.
         * @return True if the entity is valid and contains [InternodeInfo] and [Internode], false otherwise.
         */
        bool InternodeCheck(const Entity &target);
/**
         * Check if the entity is valid root.
         * @param target Target for check.
         * @return True if the entity is valid and contains [RootInfo] and [Root], false otherwise.
         */
        bool RootCheck(const Entity &target);

        /**
         * Check if the entity is valid branch.
         * @param target Target for check.
         * @return True if the entity is valid and contains [BranchInfo] and [Branch], false otherwise.
         */
        bool BranchCheck(const Entity &target);
        /**
         * The GUI menu template for creating an specific kind of internode.
         * @tparam T The target parameter of the internode.
         * @param menuTitle The title of the menu, the the button to start the menu.
         * @param parameterExtension The extension of the parameter file.
         * @param parameterInspector User-configurable GUI for inspecting the parameter.
         * @param parameterDeserializer The deserializer of the parameter.
         * @param parameterSerializer The serializer of the parameter.
         * @param internodeCreator User-configurable internode creation function with parameters given.
         */
        template<typename T>
        void CreateInternodeMenu(const std::string &menuTitle,
                                 const std::string &parameterExtension,
                                 const std::function<void(T &params)> &parameterInspector,
                                 const std::function<void(T &params,
                                                          const std::filesystem::path &path)> &parameterDeserializer,
                                 const std::function<void(const T &params,
                                                          const std::filesystem::path &path)> &parameterSerializer,
                                 const std::function<void(const T &params,
                                                            const Transform &transform)> &internodeCreator
        );

    };

    template<typename T>
    void InternodeLayer::PushInternodeBehaviour(const std::shared_ptr<T> &behaviour) {
        if (!behaviour.get()) return;
        bool search = false;
        for (auto &i: m_internodeBehaviours) {
            if (i.Get<IPlantBehaviour>()->GetTypeName() ==
                std::dynamic_pointer_cast<IPlantBehaviour>(behaviour)->GetTypeName())
                search = true;
        }
        if (!search) {
            m_internodeBehaviours.emplace_back(std::dynamic_pointer_cast<IPlantBehaviour>(behaviour));
        }
    }

    template<typename T>
    std::shared_ptr<T> InternodeLayer::GetInternodeBehaviour() {
        for (auto &i: m_internodeBehaviours) {
            if (i.Get<IPlantBehaviour>()->GetTypeName() == Serialization::GetSerializableTypeName<T>()) {
                return i.Get<T>();
            }
        }
        return nullptr;
    }

    template<typename T>
    void IPlantBehaviour::CreateInternodeMenu(const std::string &menuTitle,
                                              const std::string &parameterExtension,
                                              const std::function<void(T &params)> &parameterInspector,
                                              const std::function<void(T &params,
                                                                           const std::filesystem::path &path)> &parameterDeserializer,
                                              const std::function<void(const T &params,
                                                                           const std::filesystem::path &path)> &parameterSerializer,
                                              const std::function<void(const T &params,
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
            ImGui::BeginChild("ChildL", ImVec2(300, 600), true,
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
                                        }, false);
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
            ImGui::BeginChild("ChildR", ImVec2(600, 600), true,
                              ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Parameters")) {
                    FileUtils::OpenFile(
                            "Import parameters", "Parameters", {parameterExtension},
                            [&](const std::filesystem::path &path) {
                                parameterDeserializer(newPlantParameters[currentFocusedNewTreeIndex], path);
                            }, false);

                    FileUtils::SaveFile(
                            "Export parameters", "Parameters", {parameterExtension},
                            [&](const std::filesystem::path &path) {
                                parameterSerializer(newPlantParameters[currentFocusedNewTreeIndex], path);
                            }, false);
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
                    internodeCreator(newPlantParameters[i], treeTransform);
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
    Entity IPlantBehaviour::CreateHelper(const Entity &parent) {
        Entity retVal;
        std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
        retVal = Entities::CreateEntity(Entities::GetCurrentScene(), m_internodeArchetype, "Internode");
        retVal.SetParent(parent);
        InternodeInfo internodeInfo;
        retVal.SetDataComponent(internodeInfo);
        auto parentInternode = parent.GetOrSetPrivateComponent<Internode>().lock();
        auto internode = retVal.GetOrSetPrivateComponent<Internode>().lock();
        internode->m_resource = std::make_unique<T>();
        internode->m_foliage = parentInternode->m_foliage;
        internode->m_currentRoot = parentInternode->m_currentRoot;
        return retVal;
    }
    template<typename T>
    Entity IPlantBehaviour::CreateRootHelper(Entity& rootInternode, Entity& rootBranch) {
        std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
        Entity rootEntity;
        rootEntity = Entities::CreateEntity(Entities::GetCurrentScene(), m_rootArchetype, "Root");
        RootInfo rootInfo;
        rootEntity.SetDataComponent(rootInfo);
        auto root = rootEntity.GetOrSetPrivateComponent<Root>().lock();

        rootInternode = Entities::CreateEntity(Entities::GetCurrentScene(), m_internodeArchetype, "Internode");
        rootInternode.SetParent(rootEntity);

        rootBranch = Entities::CreateEntity(Entities::GetCurrentScene(), m_branchArchetype, "Branch");
        rootBranch.SetParent(rootEntity);

        InternodeInfo internodeInfo;
        rootInternode.SetDataComponent(internodeInfo);
        auto internode = rootInternode.GetOrSetPrivateComponent<Internode>().lock();
        internode->m_resource = Serialization::ProduceSerializable<T>();
        internode->m_foliage = AssetManager::CreateAsset<InternodeFoliage>("Foliage");
        internode->m_currentRoot = rootEntity;
        BranchInfo branchInfo;
        rootBranch.SetDataComponent(branchInfo);
        auto branch = rootBranch.GetOrSetPrivateComponent<Branch>().lock();
        branch->m_currentRoot = rootEntity;
        return rootEntity;
    }




}