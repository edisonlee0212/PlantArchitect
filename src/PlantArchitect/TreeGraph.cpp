#include "TreeGraph.hpp"
#include "GeneralTreeBehaviour.hpp"
#include "InternodeLayer.hpp"
#include "SpaceColonizationBehaviour.hpp"
using namespace PlantArchitect;

void TreeGraph::Serialize(YAML::Emitter& out) {
    out << YAML::Key << "name" << YAML::Value << m_name;
    out << YAML::Key << "layersize" << YAML::Value << m_layerSize;
    out << YAML::Key << "layers" << YAML::Value << YAML::BeginMap;
    std::vector<std::vector<std::shared_ptr<TreeGraphNode>>> graphNodes;
    graphNodes.resize(m_layerSize);
    CollectChild(m_root, graphNodes, 0);
    for (int layerIndex = 0; layerIndex < m_layerSize; layerIndex++) {
        out << YAML::Key << std::to_string(layerIndex) << YAML::Value << YAML::BeginMap;
        {
            auto& layer = graphNodes[layerIndex];
            out << YAML::Key << "internodesize" << YAML::Value << layer.size();
            for (int nodeIndex = 0; nodeIndex < layer.size(); nodeIndex++) {
                auto node = layer[nodeIndex];
                out << YAML::Key << std::to_string(nodeIndex) << YAML::Value << YAML::BeginMap;
                {
                    out << YAML::Key << "id" << YAML::Value << node->m_id;
                    out << YAML::Key << "parent" << YAML::Value << node->m_parentId;
                    out << YAML::Key << "quat" << YAML::Value << YAML::BeginSeq;
                    for (int i = 0; i < 4; i++) {
                        out << YAML::BeginMap;
                        out << std::to_string(node->m_globalRotation[i]);
                        out << YAML::EndMap;
                    }
                    out << YAML::EndSeq;

                    out << YAML::Key << "position" << YAML::Value << YAML::BeginSeq;
                    for (int i = 0; i < 3; i++) {
                        out << YAML::BeginMap;
                        out << std::to_string(node->m_position[i]);
                        out << YAML::EndMap;
                    }
                    out << YAML::EndSeq;

                    out << YAML::Key << "thickness" << YAML::Value << node->m_thickness;
                    out << YAML::Key << "length" << YAML::Value << node->m_length;
                }
                out << YAML::EndMap;
            }
        }
        out << YAML::EndMap;
    }

    out << YAML::EndMap;
}

Entity TreeGraph::InstantiateTree() {
    if (!m_root) return {};
    
    auto scene = Application::GetActiveScene();
    Entity rootInternode, rootBranch;
    AssetRef ref;
    std::shared_ptr<IPlantBehaviour> behaviour;
    if(m_plantDescriptor.Get<GeneralTreeParameters>())
    {
        ref = m_plantDescriptor;
        behaviour = std::dynamic_pointer_cast<IPlantBehaviour>(Application::GetLayer<InternodeLayer>()->GetPlantBehaviour<GeneralTreeBehaviour>());
    }
    else if(m_plantDescriptor.Get<SpaceColonizationParameters>())
    {
        ref = m_plantDescriptor;
        behaviour = std::dynamic_pointer_cast<IPlantBehaviour>(Application::GetLayer<InternodeLayer>()->GetPlantBehaviour<SpaceColonizationBehaviour>());
    }
    else
    {
        ref = ProjectManager::CreateTemporaryAsset<GeneralTreeParameters>();
        behaviour = std::dynamic_pointer_cast<IPlantBehaviour>(Application::GetLayer<InternodeLayer>()->GetPlantBehaviour<GeneralTreeBehaviour>());
    }
	auto root = behaviour->CreateRoot(scene, ref, rootInternode, rootBranch);
    scene->SetEntityName(root, m_name);
    InternodeInfo rootInternodeInfo;
    rootInternodeInfo.m_thickness = m_root->m_thickness;
    rootInternodeInfo.m_length = m_root->m_length;
    GlobalTransform rootGlobalTransform;
    rootGlobalTransform.SetRotation(m_root->m_globalRotation);
    rootGlobalTransform.SetPosition(m_root->m_start);
    scene->SetDataComponent(rootInternode, rootGlobalTransform);
    scene->SetDataComponent(rootInternode, rootInternodeInfo);

    InstantiateChildren(scene, behaviour, rootInternode, m_root, rootInternodeInfo.m_length);

    scene->ForEach<GlobalTransform, Transform, InternodeInfo, InternodeStatus>
        (Jobs::Workers(), behaviour->m_internodesQuery,
            [&](int i, Entity entity,
                GlobalTransform& globalTransform,
                Transform& transform,
                InternodeInfo& internodeInfo, InternodeStatus& internodeStatus) {
                    auto parent = scene->GetParent(entity);
    if (parent.GetIndex() == 0) {
        internodeInfo.m_localRotation = globalTransform.GetRotation();
    }
    else {
        auto parentGlobalTransform = scene->GetDataComponent<GlobalTransform>(parent);
        auto parentGlobalRotation = parentGlobalTransform.GetRotation();
        internodeInfo.m_localRotation =
            glm::inverse(parentGlobalRotation) * globalTransform.GetRotation();
    }
    auto childAmount = scene->GetChildrenAmount(entity);
    internodeInfo.m_endNode = childAmount == 0;
    auto internode = scene->GetOrSetPrivateComponent<Internode>(entity).lock();
            }
    );

    Application::GetLayer<InternodeLayer>()->Preprocess(scene);

    return root;
}

void TreeGraph::CollectAssetRef(std::vector<AssetRef>& list)
{
    if(m_plantDescriptor.Get<IAsset>()) list.push_back(m_plantDescriptor);
}

void TreeGraph::InstantiateChildren(const std::shared_ptr<Scene>& scene,
                                    const std::shared_ptr<IPlantBehaviour>& behaviour, const Entity& parent,
                                    const std::shared_ptr<TreeGraphNode>& node, float currentLength) const {
    bool childExist = false;
    int childCount = 0;
	for (const auto& childNode : node->m_children) {
        auto child = behaviour->CreateInternode(scene, parent);
        scene->GetOrSetPrivateComponent<Internode>(child).lock()->m_fromApicalBud = !childExist;
        InternodeInfo internodeInfo;
        internodeInfo.m_thickness = childNode->m_thickness;
        internodeInfo.m_length = childNode->m_length;
        GlobalTransform globalTransform;
        globalTransform.SetRotation(childNode->m_globalRotation);
        globalTransform.SetPosition(childNode->m_start);
        scene->SetDataComponent(child, globalTransform);
        scene->SetDataComponent(child, internodeInfo);

        if(scene->HasDataComponent<InternodeStatus>(child))
        {
            InternodeStatus status;
            status.m_branchingOrder = childCount;
            status.m_desiredLocalRotation = glm::inverse(scene->GetDataComponent<GlobalTransform>(parent).GetRotation()) * childNode->m_globalRotation;
            status.m_branchLength = childNode->m_length;
            scene->SetDataComponent(child, status);
        }

        if (!m_enableInstantiateLengthLimit || currentLength + internodeInfo.m_length < m_instantiateLengthLimit) {
            InstantiateChildren(scene, behaviour, child, childNode, currentLength + internodeInfo.m_length);
        }
        childExist = true;
    }
    if(childExist)
    {
        scene->GetOrSetPrivateComponent<Internode>(parent).lock()->m_apicalBud.m_status = BudStatus::Died;
    }
}

struct GraphNode {
    int m_id;
    int m_parent;
    glm::vec3 m_position;
    glm::vec3 m_direction;
    float m_radius;
    float m_length;
};

void TreeGraph::Deserialize(const YAML::Node& in) {
    m_saved = true;
    m_name = GetTitle();
    int id = 0;
    std::vector<GraphNode> nodes;
    while (in[std::to_string(id)]) {
        auto& inNode = in[std::to_string(id)];
        nodes.emplace_back();
        auto& node = nodes.back();
        node.m_id = id;
        node.m_parent = inNode["parent"].as<int>();
        int index = 0;
        for (const auto& component : inNode["position"]) {
            node.m_position[index] = component.as<float>();
            index++;
        }
        index = 0;
        for (const auto& component : inNode["direction"]) {
            node.m_direction[index] = component.as<float>();
            index++;
        }
        node.m_radius = inNode["radius"].as<float>();
        node.m_length = inNode["length"].as<float>();
        id++;
    }
    std::unordered_map<int, std::shared_ptr<TreeGraphNode>> previousNodes;
    m_root = std::make_shared<TreeGraphNode>();
    m_root->m_start = glm::vec3(0, 0, 0);
    m_root->m_length = nodes[0].m_length;
    m_root->m_thickness = nodes[0].m_radius;
    m_root->m_id = 0;
    m_root->m_parentId = -1;
    m_root->m_fromApicalBud = true;
    m_root->m_globalRotation = glm::quatLookAt(nodes[0].m_direction, glm::vec3(nodes[0].m_direction.y, nodes[0].m_direction.z,
        nodes[0].m_direction.x)
        );
    previousNodes[0] = m_root;
    for (id = 1; id < nodes.size(); id++) {
        auto& node = nodes[id];
        auto parentNodeId = node.m_parent;
        auto& parentNode = previousNodes[parentNodeId];
        auto newNode = std::make_shared<TreeGraphNode>();
        newNode->m_id = id;
        newNode->m_start = parentNode->m_start + parentNode->m_length *
            (glm::normalize(parentNode->m_globalRotation) *
                glm::vec3(0, 0, -1));
        newNode->m_thickness = node.m_radius;
        newNode->m_length = node.m_length;
        newNode->m_parentId = parentNodeId;
        newNode->m_position = node.m_position;
        newNode->m_globalRotation = glm::quatLookAt(node.m_direction, glm::vec3(node.m_direction.y, node.m_direction.z,
            node.m_direction.x)
        );
        if (parentNode->m_children.empty()) newNode->m_fromApicalBud = true;
        previousNodes[id] = newNode;
        parentNode->m_children.push_back(newNode);
        newNode->m_parent = parentNode;
    }
#ifdef OLD_GRAPH
    m_name = in["name"].as<std::string>();
    m_layerSize = in["layersize"].as<int>();
    auto layers = in["layers"];
    auto rootLayer = layers["0"];
    std::unordered_map<int, std::shared_ptr<TreeGraphNode>> previousNodes;
    m_root = std::make_shared<TreeGraphNode>();
    m_root->m_start = glm::vec3(0, 0, 0);
    int rootIndex = 0;
    auto rootNode = rootLayer["0"];
    m_root->m_length = rootNode["length"].as<float>();
    m_root->m_thickness = rootNode["thickness"].as<float>();
    m_root->m_id = rootNode["id"].as<int>();
    m_root->m_parentId = -1;
    m_root->m_fromApicalBud = true;
    int index = 0;
    for (const auto& component : rootNode["quat"]) {
        m_root->m_globalRotation[index] = component.as<float>();
        index++;
    }
    index = 0;
    for (const auto& component : rootNode["position"]) {
        m_root->m_position[index] = component.as<float>();
        index++;
    }
    previousNodes[m_root->m_id] = m_root;
    for (int layerIndex = 1; layerIndex < m_layerSize; layerIndex++) {
        auto layer = layers[std::to_string(layerIndex)];
        auto internodeSize = layer["internodesize"].as<int>();
        for (int nodeIndex = 0; nodeIndex < internodeSize; nodeIndex++) {
            auto node = layer[std::to_string(nodeIndex)];
            auto parentNodeId = node["parent"].as<int>();
            if (parentNodeId == -1) parentNodeId = 0;
            auto& parentNode = previousNodes[parentNodeId];
            auto newNode = std::make_shared<TreeGraphNode>();
            newNode->m_id = node["id"].as<int>();
            newNode->m_start = parentNode->m_start + parentNode->m_length *
                (glm::normalize(parentNode->m_globalRotation) *
                    glm::vec3(0, 0, -1));
            newNode->m_thickness = node["thickness"].as<float>();
            newNode->m_length = node["length"].as<float>();
            newNode->m_parentId = parentNodeId;
            if (newNode->m_parentId == 0) newNode->m_parentId = -1;
            index = 0;
            for (const auto& component : node["quat"]) {
                newNode->m_globalRotation[index] = component.as<float>();
                index++;
            }
            index = 0;
            for (const auto& component : node["position"]) {
                newNode->m_position[index] = component.as<float>();
                index++;
            }
            if (parentNode->m_children.empty()) newNode->m_fromApicalBud = true;
            previousNodes[newNode->m_id] = newNode;
            parentNode->m_children.push_back(newNode);
            newNode->m_parent = parentNode;
        }
    }
#endif

}

void TreeGraph::OnInspect()
{
    ImGui::Checkbox("Length limit", &m_enableInstantiateLengthLimit);
    ImGui::DragFloat("Length limit", &m_instantiateLengthLimit, 0.1f);
	IPlantDescriptor::OnInspect();

    Editor::DragAndDropButton(m_plantDescriptor, "Parameters",
        { "GeneralTreeParameters", "SpaceColonizationParameters"}, true);
}

void TreeGraph::CollectChild(const std::shared_ptr<TreeGraphNode>& node,
                             std::vector<std::vector<std::shared_ptr<TreeGraphNode>>>& graphNodes,
                             int currentLayer) const {
    graphNodes[currentLayer].push_back(node);
    for (const auto& i : node->m_children) {
        CollectChild(i, graphNodes, currentLayer + 1);
    }
}
