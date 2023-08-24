#pragma once

#include "RingSegment.hpp"
#include "plant_architect_export.h"
#include "InternodeModel/InternodeResources/IInternodeResource.hpp"
#include "DataComponents.hpp"
#include "TreeIO.hpp"

using namespace treeio;
using namespace UniEngine;
namespace PlantArchitect {
    struct LSystemString;
    enum class PLANT_ARCHITECT_API BudStatus {
        Sleeping,
        Flushing,
        Flushed,
        Died
    };

    struct PLANT_ARCHITECT_API Bud : public ISerializable {
        BudStatus m_status = BudStatus::Sleeping;
        InternodeInfo m_newInternodeInfo;
        float m_flushProbability = 0;

        void OnInspect();

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void Save(const std::string &name, YAML::Emitter &out);

        void Load(const std::string &name, const YAML::Node &in);
    };

    struct LSystemCommand;


    class PLANT_ARCHITECT_API Internode : public IPrivateComponent {
        void ExportLSystemCommandsHelper(int &index, const Entity &target, std::vector<LSystemCommand> &commands);

        void CollectInternodesHelper(const Entity &target, std::vector<Entity> &results);

        void ExportTreeIOTreeHelper(ArrayTree &tree, const Entity &target,
                                    ArrayTreeT<TreeNodeData, TreeMetaData>::NodeIdT id);

        friend class IPlantBehaviour;

        /**
         * Normal direction for mesh generation
         */
        glm::vec3 m_normalDir = glm::vec3(0, 0, 1);
        /**
         * Subdivision step for mesh generation
         */
        int m_step = 4;


    public:
        std::vector<glm::vec4> m_twigAnchors;
        /**
             * For mesh generation
             */
        std::vector<RingSegment> m_rings;
        /**
         * The current root of the internode.
         */
        EntityRef m_currentRoot;

        /**
         * The generated matrices of foliage
         */
        std::vector<glm::mat4> m_foliageMatrices;

        /**
         * Whether this internode is formed from an apical bud
         */
        bool m_fromApicalBud;

        /**
         * The resource storage for the internode.
         */
        std::shared_ptr<IInternodeResource> m_resource;
        /**
         * The apical bud.
         */
        Bud m_apicalBud;
        /**
         * The axillary or lateral bud.
         */
        std::vector<Bud> m_lateralBuds;

        /**
         * Collect all subsequent internodes from this internode.
         * @param results a list of internodes as descendents
         */
        void CollectInternodes(std::vector<Entity> &results);

        /**
         * Collect resource (auxin, nutrients, etc.)
         * @param deltaTime How much time the action takes.
         */
        void CollectResource(float deltaTime);

        void Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) override;

        void OnDestroy() override;

        /**
         * Down stream the resources.
         * @param deltaTime
         */
        void DownStreamResource(float deltaTime);

        /**
         * Up stream the resources.
         * @param deltaTime How much time the action takes.
         */
        void UpStreamResource(float deltaTime);

        void OnCreate() override;

        /*
         * Parse the structure of the internodes and set up commands.
         */
        void ExportLString(const std::shared_ptr<LSystemString> &lString);

        /*
         * Parse the structure of the internodes and set up commands.
         */
        void ExportTreeIOTree(const std::filesystem::path &path);

        void OnInspect() override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;


        void PostCloneAction(const std::shared_ptr<IPrivateComponent> &target) override;

        void Serialize(YAML::Emitter &out) override;

        [[nodiscard]] Bound CalculateChildrenBound();

        void Deserialize(const YAML::Node &in) override;


    };

    template<typename T>
    void SaveList(const std::string &name, std::vector<T> &target, YAML::Emitter &out) {
        if (target.empty()) return;
        out << YAML::Key << name << YAML::Value << YAML::BeginSeq;
        for (auto &i: target) {
            out << YAML::BeginMap;
            static_cast<ISerializable *>(&i)->Serialize(out);
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
    }

    template<typename T>
    void LoadList(const std::string &name, std::vector<T> target, const YAML::Node &in) {
        if (in[name]) {
            target.clear();
            for (const auto &i: in[name]) {
                T instance;
                static_cast<ISerializable *>(&instance)->Deserialize(i);
                target.push_back(instance);
            }
        }
    }

    template<typename T>
    void SaveListAsBinary(const std::string &name, const std::vector<T> &target, YAML::Emitter &out) {
        if (!target.empty()) {
            out << YAML::Key << name << YAML::Value
                << YAML::Binary((const unsigned char *) target.data(), target.size() * sizeof(T));
        }
    }

    template<typename T>
    void LoadListFromBinary(const std::string &name, std::vector<T> target, const YAML::Node &in) {
        if (in[name]) {
            YAML::Binary binaryList = in[name].as<YAML::Binary>();
            target.resize(binaryList.size() / sizeof(T));
            std::memcpy(target.data(), binaryList.data(), binaryList.size());
        }
    }
}