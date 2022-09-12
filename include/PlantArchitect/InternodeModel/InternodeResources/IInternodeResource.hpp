#pragma once
#include <plant_architect_export.h>
using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API IInternodeResource : public ISerializable{
    public:
        /**
         * Define how to collect resource from environment.
         * @param deltaTime The time passed for the action.
         * @param self The internode that collect the resource.
         */
        virtual void Collect(float deltaTime, const Entity& self) = 0;
        /**
         * Define how to pass the resource (end node to root).
         * @param deltaTime Time passed in this action.
         * @param self The internode that streams the resource.
         * @param target The internode that receive the resource.
         */
        virtual void DownStream(float deltaTime, const Entity& self, const Entity& target) = 0;
        /**
         * Define how to pass the resource (root to end node).
         * @param deltaTime Time passed in this action.
         * @param self The internode that streams the resource.
         * @param target The internode that receive the resource.
         */
        virtual void UpStream(float deltaTime, const Entity& self, const Entity& target) = 0;
        /**
         * Clear all resources.
         */
        virtual void Reset() = 0;
    };
}