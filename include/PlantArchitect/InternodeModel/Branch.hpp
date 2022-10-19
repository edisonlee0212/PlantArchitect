#pragma once

#include "RingSegment.hpp"
#include "plant_architect_export.h"
#include "InternodeModel/InternodeResources/IInternodeResource.hpp"
#include "DataComponents.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Branch : public IPrivateComponent {
    public:
        std::vector<Entity> m_internodeChain;

        /**
         * The current root of the internode.
         */
        EntityRef m_currentRoot;
/**
         * The current root of the internode.
         */
        EntityRef m_currentInternode;


        void Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) override;
        void OnDestroy();
        void OnInspect() override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}