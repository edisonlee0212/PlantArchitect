#pragma once

#include <plant_architect_export.h>
#include <IInternodeBehaviour.hpp>
using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API JSONTreeTag : public IDataComponent {
    };

    struct PLANT_ARCHITECT_API JSONTreeParameters : public IDataComponent {
        float m_internodeLength = 1.0f;
        float m_thicknessFactor = 0.5f;
        float m_endNodeThickness = 0.02f;

        void OnInspect();
    };

    //This class is the bridge between the raw JSON file and the data you design which you will use it to formulate the tree structure.
    class PLANT_ARCHITECT_API JSONData : public IAsset {
    protected:
        bool SaveInternal(const std::filesystem::path &path) override;

        bool LoadInternal(const std::filesystem::path &path) override;

    public:
        void ParseJSONString(const std::string &string);

        //TODO: Declare data structure for your needs.
    };

    class PLANT_ARCHITECT_API JSONTreeBehaviour : public IInternodeBehaviour {
    protected:
        bool InternalInternodeCheck(const Entity &target) override;
    public:
        Entity FormPlant(const std::shared_ptr<JSONData> &lString, const JSONTreeParameters &parameters);
        void OnInspect() override;

        void OnCreate() override;

        Entity Retrieve() override;

        Entity Retrieve(const Entity &parent) override;
    };
}