#pragma once
#include <plant_architect_export.h>
#include <Internode.hpp>
using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API InternodeTag : public IDataComponent {

    };
    class PLANT_ARCHITECT_API InternodeSystem : public ISystem {
    public:
        void Simulate(float deltaTime);
        void OnCreate() override;
        void OnInspect() override;
    private:
        /**
         * The EntityQuery for filtering all internodes.
         */
        EntityQuery m_internodesQuery;
        friend class Internode;

        int m_internodeBehavioursSize = 0;
        std::vector<AssetRef> m_internodeBehaviours;
        void BehaviourSlotButton();
    };
}