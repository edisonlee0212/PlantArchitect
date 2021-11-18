//
// Created by lllll on 11/17/2021.
//

#include <EmptyInternodeResource.hpp>
#include "JSONTreeBehaviour.hpp"
using namespace PlantArchitect;

bool JSONTreeBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<JSONTreeTag>();
}

void JSONTreeBehaviour::OnInspect() {
    RecycleButton();
    static JSONTreeParameters parameters;
    parameters.OnInspect();
    //button here
    static AssetRef tempStoredJSONData;
    EditorManager::DragAndDropButton<JSONData>(tempStoredJSONData, "Create tree from JSON Data here: ");
    auto jSONData = tempStoredJSONData.Get<JSONData>();
    if (jSONData) FormPlant(jSONData, parameters);
    tempStoredJSONData.Clear();

    static float resolution = 0.02;
    static float subdivision = 4.0;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate meshes")) {
        GenerateSkinnedMeshes(subdivision, resolution);
    }
}

void JSONTreeBehaviour::OnCreate() {
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("JSON Tree Internode", InternodeInfo(),
                                                 JSONTreeTag(), JSONTreeParameters(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer(), BranchPhysicsParameters());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(JSONTreeTag());
}

Entity JSONTreeBehaviour::Retrieve() {
    auto retVal = RetrieveHelper<EmptyInternodeResource>();
    retVal.SetDataComponent(BranchPhysicsParameters());
    return retVal;
}

Entity JSONTreeBehaviour::Retrieve(const Entity &parent) {
    auto retVal = RetrieveHelper<EmptyInternodeResource>(parent);
    retVal.SetDataComponent(parent.GetDataComponent<BranchPhysicsParameters>());
    return retVal;
}

void JSONTreeParameters::OnInspect() {
    ImGui::DragFloat("Internode Length", &m_internodeLength);
    ImGui::DragFloat("Thickness Factor", &m_thicknessFactor);
    ImGui::DragFloat("End node thickness", &m_endNodeThickness);
}

bool JSONData::SaveInternal(const std::filesystem::path &path) {
    if (path.extension().string() == ".json") {
        std::ofstream of;
        of.open(path.c_str(),
                std::ofstream::out | std::ofstream::trunc);
        if (of.is_open()) {
            std::string output;
            //TODO: (Optional) Save JSON Data to json file here. This this not necessary because you can save the tree structure to L-String instead.
            of.write(output.c_str(), output.size());
            of.flush();
        }
        return true;
    }
    return false;
}

bool JSONData::LoadInternal(const std::filesystem::path &path) {
    if (path.extension().string() == ".json" || path.extension().string() == ".txt") {
        auto string = FileUtils::LoadFileAsString(path);
        ParseJSONString(string);
        return true;
    }
    return false;
}

void JSONData::ParseJSONString(const std::string &string) {
    //TODO: Parse JSON file to the meaningful data structure here. You may use external library.
}

Entity JSONTreeBehaviour::FormPlant(const std::shared_ptr<JSONData> &lString, const JSONTreeParameters &parameters) {
    Entity root;

    //TODO: Create the tree from JSONData asset here.

    //Calculate other properties like thickness after the structure of the tree is ready.
    TreeGraphWalkerEndToRoot(root, root, [&](Entity parent) {
        float thicknessCollection = 0.0f;
        auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
        auto parameters = parent.GetDataComponent<JSONTreeParameters>();
        parent.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
            if (!InternodeCheck(child)) return;
            auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
            thicknessCollection += glm::pow(childInternodeInfo.m_thickness,
                                            1.0f / parameters.m_thicknessFactor);
        });
        parentInternodeInfo.m_thickness = glm::pow(thicknessCollection, parameters.m_thicknessFactor);
        parent.SetDataComponent(parentInternodeInfo);
    }, [](Entity endNode) {
        auto internodeInfo = endNode.GetDataComponent<InternodeInfo>();
        auto parameters = endNode.GetDataComponent<JSONTreeParameters>();
        internodeInfo.m_thickness = parameters.m_endNodeThickness;
        endNode.SetDataComponent(internodeInfo);
    });
    return root;
}