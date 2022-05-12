#pragma once

#include "plant_architect_export.h"
#include "Internode.hpp"
#include "Root.hpp"
#include "Branch.hpp"
#include "Serialization.hpp"
#include "Graphics.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    struct SpaceColonizationParameters;

    struct PLANT_ARCHITECT_API MeshGeneratorSettings{
        float m_resolution = 0.02f;
        float m_subdivision = 4.0f;
        bool m_vertexColorOnly = false;
        bool m_enableFoliage = true;
        bool m_enableBranch = true;
        void OnInspect();
        void Save(const std::string &name, YAML::Emitter &out);
        void Load(const std::string &name, const YAML::Node &in);
    };

    class PLANT_ARCHITECT_API IPlantBehaviour {
        friend class PlantLayer;
        friend class TreeGraph;
    protected:
        std::string m_typeName;
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
        Entity CreateInternodeHelper(const std::shared_ptr<Scene>& scene, const Entity &parent);

        /**
         * Get or create an branch.
         * @param parent The parent of the branch.
         * @return An entity represent the branch.
         */
        Entity CreateBranchHelper(const std::shared_ptr<Scene>& scene, const Entity &parent, const Entity &internode);

        /**
         * Get or create a root.
         * @tparam T The type of resource that will be added to the internode.
         * @return An entity represent the root internode.
         */
        template<typename T>
        Entity CreateRootHelper(const std::shared_ptr<Scene>& scene, AssetRef descriptor, Entity &rootInternode, Entity &rootBranch);

#pragma endregion
#pragma region Helpers

        void BranchCollector(const std::shared_ptr<Scene>& scene, std::vector<Entity> &boundEntities,
                             std::vector<int> &parentIndices,
                             const int &parentIndex, const Entity &node);

        void BranchSkinnedMeshGenerator(const std::shared_ptr<Scene>& scene, std::vector<Entity> &entities,
                                        std::vector<int> &parentIndices,
                                        std::vector<SkinnedVertex> &vertices,
                                        std::vector<unsigned> &indices);

        void FoliageSkinnedMeshGenerator(const std::shared_ptr<Scene>& scene, std::vector<Entity> &entities,
                                         std::vector<int> &parentIndices,
                                         std::vector<SkinnedVertex> &vertices,
                                         std::vector<unsigned> &indices);

        void PrepareInternodeForSkeletalAnimation(const std::shared_ptr<Scene>& scene, const Entity &entity, Entity &branchMesh, Entity &foliageMesh, const MeshGeneratorSettings &settings);

#pragma endregion

        virtual bool InternalInternodeCheck(const std::shared_ptr<Scene>& scene, const Entity &target) = 0;

        virtual bool InternalRootCheck(const std::shared_ptr<Scene>& scene, const Entity &target) = 0;

        virtual bool InternalBranchCheck(const std::shared_ptr<Scene>& scene, const Entity &target) = 0;

        void UpdateBranchHelper(const std::shared_ptr<Scene>& scene, const Entity &currentBranch, const Entity &currentInternode);

    public:
        virtual void OnInspect() = 0;

        [[nodiscard]] std::string GetTypeName() const;

        void UpdateBranches(const std::shared_ptr<Scene>& scene);

        void ApplyTropism(const glm::vec3 &targetDir, float tropism, glm::vec3 &front, glm::vec3 &up);

        virtual Entity CreateRoot(const std::shared_ptr<Scene>& scene, AssetRef descriptor, Entity &rootInternode, Entity &rootBranch) = 0;

        virtual Entity CreateBranch(const std::shared_ptr<Scene>& scene, const Entity &parent, const Entity &internode) = 0;

        virtual Entity CreateInternode(const std::shared_ptr<Scene>& scene, const Entity &parent) = 0;

        /**
         * Remove the internode and all its descendents.
         * @param internode target internode to be removed
         */
        void DestroyInternode(const std::shared_ptr<Scene>& scene, const Entity &internode);

        /**
         * Remove the branch and all its descendents. The linked internodes will also be deleted.
         * @param branch target branch to be removed
         */
        void DestroyBranch(const std::shared_ptr<Scene>& scene, const Entity &branch);

        /**
         * Handle main growth here.
         */
        virtual void Grow(const std::shared_ptr<Scene>& scene, int iteration) {};

        /**
         * Generate skinned mesh for internodes.
         * @param entities
         */
        void
        GenerateSkinnedMeshes(const std::shared_ptr<Scene>& scene, const MeshGeneratorSettings &settings);


        /**
         * A helper method that traverse target plant internodes from root to end, and come back from end to root.
         * @param node Start node.
         * @param rootToEndAction The action to take during traversal from root to end. You must not delete the parent during the action.
         * @param endToRootAction The action to take during traversal from end to root. You must not delete the parent during the action.
         * @param endNodeAction The action to take for end nodes. You must not delete the end node during the action.
         */
        void InternodeGraphWalker(const std::shared_ptr<Scene>& scene, const Entity &startInternode,
                                  const std::function<void(Entity parent, Entity child)> &rootToEndAction,
                                  const std::function<void(Entity parent)> &endToRootAction,
                                  const std::function<void(Entity endNode)> &endNodeAction);

        /**
         * A helper method that traverse target plant internodes from root to end.
         * @param node Start node.
         * @param rootToEndAction The action to take during traversal from root to end.
         */
        void InternodeGraphWalkerRootToEnd(const std::shared_ptr<Scene>& scene, const Entity &startInternode,
                                           const std::function<void(Entity parent, Entity child)> &rootToEndAction);

        /**
         * A helper method that traverse target plant internodes from end to root.
         * @param node Start node.
         * @param endToRootAction The action to take during traversal from end to root. You must not delete the parent during the action.
         * @param endNodeAction The action to take for end nodes. You must not delete the end node during the action.
         */
        void InternodeGraphWalkerEndToRoot(const std::shared_ptr<Scene>& scene, const Entity &startInternode,
                                           const std::function<void(Entity parent)> &endToRootAction,
                                           const std::function<void(Entity endNode)> &endNodeAction);

        /**
         * A helper method that traverse target plant branches from root to end, and come back from end to root.
         * @param node Start node.
         * @param rootToEndAction The action to take during traversal from root to end. You must not delete the parent during the action.
         * @param endToRootAction The action to take during traversal from end to root. You must not delete the parent during the action.
         * @param endNodeAction The action to take for end. You must not delete the end node during the action.
         */
        void BranchGraphWalker(const std::shared_ptr<Scene>& scene, const Entity &startBranch,
                               const std::function<void(Entity parent, Entity child)> &rootToEndAction,
                               const std::function<void(Entity parent)> &endToRootAction,
                               const std::function<void(Entity endNode)> &endNodeAction);

        /**
         * A helper method that traverse target plant branches from root to end.
         * @param node Start node.
         * @param rootToEndAction The action to take during traversal from root to end.
         */
        void BranchGraphWalkerRootToEnd(const std::shared_ptr<Scene>& scene, const Entity &startBranch,
                                        const std::function<void(Entity parent, Entity child)> &rootToEndAction);

        /**
         * A helper method that traverse target plant branches from end to root.
         * @param node Start node.
         * @param endToRootAction The action to take during traversal from end to root. You must not delete the parent during the action.
         * @param endNodeAction The action to take for end nodes. You must not delete the end node during the action.
         */
        void BranchGraphWalkerEndToRoot(const std::shared_ptr<Scene>& scene, const Entity &startBranch,
                                        const std::function<void(Entity parent)> &endToRootAction,
                                        const std::function<void(Entity endNode)> &endNodeAction);


        /**
         * Check if the entity is valid internode.
         * @param target Target for check.
         * @return True if the entity is valid and contains [InternodeInfo] and [Internode], false otherwise.
         */
        bool InternodeCheck(const std::shared_ptr<Scene>& scene, const Entity &target);

/**
         * Check if the entity is valid root.
         * @param target Target for check.
         * @return True if the entity is valid and contains [RootInfo] and [Root], false otherwise.
         */
        bool RootCheck(const std::shared_ptr<Scene>& scene, const Entity &target);

        /**
         * Check if the entity is valid branch.
         * @param target Target for check.
         * @return True if the entity is valid and contains [BranchInfo] and [Branch], false otherwise.
         */
        bool BranchCheck(const std::shared_ptr<Scene>& scene, const Entity &target);

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
        void CreateInternodeMenu(bool &open, const std::string &menuTitle,
                                 const std::function<void(AssetRef descriptor,
                                                          const Transform &transform)> &internodeCreator
        );

    };


    template<typename T>
    void IPlantBehaviour::CreateInternodeMenu(bool &open, const std::string &menuTitle,
                                              const std::function<void(AssetRef descriptor,
                                                                       const Transform &transform)> &internodeCreator
    ) {
        if (ImGui::Begin(menuTitle.c_str(), &open)) {
            static std::vector<AssetRef> newPlantParameters;
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
                    if (Editor::DragAndDropButton<T>(newPlantParameters[0], "Descriptors for all", false)) {
                        for (int i = 1; i < newPlantParameters.size(); i++)
                            newPlantParameters[i] = newPlantParameters[0];
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
            ImGui::Columns(1);
            if (newTreePositions.size() < newTreeAmount) {
                if (newPlantParameters.empty()) {
                    newPlantParameters.resize(1);
                    AssetRef descriptor = ProjectManager::CreateTemporaryAsset<T>();
                    newPlantParameters[0] = descriptor;
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
                    Editor::DragAndDropButton<T>(newPlantParameters[currentFocusedNewTreeIndex], "Descriptor",
                                                 false);
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
            ImGui::Columns(1);
            ImGui::PushItemWidth(200);
            newPlantParameters[currentFocusedNewTreeIndex].Get<IPlantDescriptor>()->OnInspect();
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
            }
        }
    }

    template<typename T>
    Entity IPlantBehaviour::CreateInternodeHelper(const std::shared_ptr<Scene>& scene, const Entity &parent) {
        Entity retVal;
        std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
        retVal = scene->CreateEntity(m_internodeArchetype, "Internode");
        scene->SetParent(retVal, parent);
        InternodeInfo internodeInfo;
        scene->SetDataComponent(retVal, internodeInfo);
        auto parentInternode = scene->GetOrSetPrivateComponent<Internode>(parent).lock();
        auto internode = scene->GetOrSetPrivateComponent<Internode>(retVal).lock();
        internode->m_resource = std::make_unique<T>();
        internode->m_currentRoot = parentInternode->m_currentRoot;
        return retVal;
    }

    template<typename T>
    Entity IPlantBehaviour::CreateRootHelper(const std::shared_ptr<Scene>& scene, AssetRef descriptor, Entity &rootInternode, Entity &rootBranch) {
        if (!descriptor.Get<IPlantDescriptor>()) {
            UNIENGINE_ERROR("Descriptor invalid!");
            return {};
        }
        std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
        Entity rootEntity;
        rootEntity = scene->CreateEntity(m_rootArchetype, "Root");
        RootInfo rootInfo;
        scene->SetDataComponent(rootEntity, rootInfo);
        auto root = scene->GetOrSetPrivateComponent<Root>(rootEntity).lock();
        root->m_plantDescriptor = descriptor;
        rootInternode = scene->CreateEntity(m_internodeArchetype, "Internode");
        scene->SetParent(rootInternode, rootEntity);

        rootBranch = scene->CreateEntity(m_branchArchetype, "Branch");
        scene->SetParent(rootBranch, rootEntity);

        InternodeInfo internodeInfo;
        scene->SetDataComponent(rootInternode, internodeInfo);
        auto internode = scene->GetOrSetPrivateComponent<Internode>(rootInternode).lock();
        internode->m_resource = Serialization::ProduceSerializable<T>();
        internode->m_currentRoot = rootEntity;
        BranchInfo branchInfo;
        branchInfo.m_endNode = true;
        scene->SetDataComponent(rootBranch, branchInfo);
        auto branch = scene->GetOrSetPrivateComponent<Branch>(rootBranch).lock();
        branch->m_currentRoot = rootEntity;
        branch->m_currentInternode = rootInternode;
        return rootEntity;
    }


}