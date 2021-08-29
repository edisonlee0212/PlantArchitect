#pragma once

#include <plant_architect_export.h>

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
        Entity Retrieve();
        void Recycle(const Entity &internode);
        void RecycleSingle(const Entity &internode);
#pragma endregion
    public:
        /**
         * What to do before the growth, and before the resource collection. Mesh, graph calculation...
         */
        virtual void PreProcess() = 0;

        /**
         * Handle main growth here.
         */
        virtual void Grow() = 0;

        /**
         * What to do after the growth. Mesh, graph calculation...
         */
        virtual void PostProcess() = 0;
    };
}