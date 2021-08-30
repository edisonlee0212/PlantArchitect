#pragma once

#include <plant_architect_export.h>
#include <Internode.hpp>
using namespace UniEngine;
namespace PlantArchitect {
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
        virtual void GenerateBranchSkinnedMeshes(const std::vector<Entity> &entities);
    };


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