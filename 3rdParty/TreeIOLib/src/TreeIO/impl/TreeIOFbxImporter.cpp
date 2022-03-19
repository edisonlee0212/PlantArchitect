/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Importer for FBX meshes and skeletons.
 */

#include "TreeIOFbxImporter.h"

#include <fstream>
#include <iostream>
#include <streambuf>
#include <sstream>
#include <stack>

#include "openfbx/ofbx.h"

#include "TreeIOVectorT.h"

namespace treeio
{

namespace impl
{

/// @brief Pack of attributes for one model.
struct MeshPack
{
    std::vector<ModelImporterBase::VertexElementT> &vertices;
    std::vector<ModelImporterBase::VertexElementT> &texCoords;
    std::vector<ModelImporterBase::VertexElementT> &normals;
    std::vector<ModelImporterBase::VertexElementT> &tangents;
    std::vector<ModelImporterBase::VertexElementT> &bitangents;
    std::vector<ModelImporterBase::VertexElementT> &colors;
    std::vector<ModelImporterBase::IndexElementT> &indices;
    std::size_t &vertexCount;
    std::size_t &indexCount;
}; // struct MeshPack

/// @brief Load all meshes from the input scene into given buffers.
void loadMeshes(const ofbx::IScene &scene, const MeshPack &pack);

/// @brief Load one mesh from the input scene into given buffers.
void loadMesh(const ofbx::IScene &scene, std::size_t meshIdx, const MeshPack &pack);

/// @brief Load skeleton from the input scene into given tree.
void loadSkeleton(const ofbx::IScene &scene, ArrayTree &skeleton);

} // namespace impl

namespace impl
{

void loadMeshes(const ofbx::IScene &scene, const MeshPack &pack)
{
    const auto meshCount{ std::max<std::size_t>(0u, scene.getMeshCount()) };
    for (std::size_t meshIdx = 0u; meshIdx < meshCount; ++meshIdx)
    { loadMesh(scene, meshIdx, pack); }
}

void loadMesh(const ofbx::IScene &scene, std::size_t meshIdx, const MeshPack &pack)
{
    if (meshIdx >= static_cast<std::size_t>(scene.getMeshCount()))
    { return; }

    const auto &mesh{ *scene.getMesh(static_cast<int>(meshIdx)) };
    const auto &geometry{ *mesh.getGeometry() };

    const auto &meshVertices{ geometry.getVertices() };
    const auto &meshTexCoords{ geometry.getUVs() };
    const auto &meshNormals{ geometry.getNormals() };
    const auto &meshTangents{ geometry.getTangents() };
    const auto &meshColors{ geometry.getColors() };
    const auto &meshIndices{ geometry.getFaceIndices() };

    const auto vertexCount{ geometry.getVertexCount() >= 0 ? static_cast<std::size_t>(geometry.getVertexCount()) : 0u };
    const auto indexCount{ geometry.getIndexCount() >= 0 ? static_cast<std::size_t>(geometry.getIndexCount()) : 0u };

    const auto originalVertexCount{ pack.vertices.size() / ModelImporterBase::COORDINATE_ELEMENTS };
    const auto originalIndexCount{ pack.colors.size() };

    // Pre-allocate data arrays.
    const auto newTotalVertexCount{ originalVertexCount + vertexCount };
    if (meshVertices)
    { pack.vertices.resize(newTotalVertexCount * ModelImporterBase::COORDINATE_ELEMENTS); }
    if (meshTexCoords)
    { pack.texCoords.resize(newTotalVertexCount * ModelImporterBase::TEX_COORD_ELEMENTS); }
    if (meshNormals)
    { pack.normals.resize(newTotalVertexCount * ModelImporterBase::NORMAL_ELEMENTS); }
    if (meshTangents)
    { pack.tangents.resize(newTotalVertexCount * ModelImporterBase::TANGENT_ELEMENTS); }
    if (meshNormals && meshTangents)
    { pack.bitangents.resize(newTotalVertexCount * ModelImporterBase::BITANGENT_ELEMENTS); }
    if (meshColors)
    { pack.colors.resize(newTotalVertexCount * ModelImporterBase::COLOR_ELEMENTS); }

    // Copy vertex data:
    for (std::size_t vIdx = 0u; vIdx < vertexCount; ++vIdx)
    { // Process all vertices:
        const auto fullIdx{ vIdx + originalVertexCount };

        // Positions:
        if (meshVertices)
        {
            const auto &position{ meshVertices[vIdx] };
            pack.vertices[fullIdx * ModelImporterBase::COORDINATE_ELEMENTS + 0u] = static_cast<float>(position.x);
            pack.vertices[fullIdx * ModelImporterBase::COORDINATE_ELEMENTS + 1u] = static_cast<float>(position.y);
            pack.vertices[fullIdx * ModelImporterBase::COORDINATE_ELEMENTS + 2u] = static_cast<float>(position.z);
        }

        // Texture coordinates:
        if (meshTexCoords)
        {
            const auto uv{ meshTexCoords[vIdx] };
            pack.texCoords[fullIdx * ModelImporterBase::TEX_COORD_ELEMENTS + 0u] = static_cast<float>(uv.x);
            pack.texCoords[fullIdx * ModelImporterBase::TEX_COORD_ELEMENTS + 1u] = static_cast<float>(uv.y);
        }

        // Normals:
        if (meshNormals)
        {
            const auto normal{ meshNormals[vIdx] };
            pack.normals[fullIdx * ModelImporterBase::NORMAL_ELEMENTS + 0u] = static_cast<float>(normal.x);
            pack.normals[fullIdx * ModelImporterBase::NORMAL_ELEMENTS + 1u] = static_cast<float>(normal.y);
            pack.normals[fullIdx * ModelImporterBase::NORMAL_ELEMENTS + 2u] = static_cast<float>(normal.z);
        }

        // Tangents:
        if (meshTangents)
        {
            const auto tangent{ meshTangents[vIdx] };
            pack.tangents[fullIdx * ModelImporterBase::TANGENT_ELEMENTS + 0u] = static_cast<float>(tangent.x);
            pack.tangents[fullIdx * ModelImporterBase::TANGENT_ELEMENTS + 1u] = static_cast<float>(tangent.y);
            pack.tangents[fullIdx * ModelImporterBase::TANGENT_ELEMENTS + 2u] = static_cast<float>(tangent.z);
            pack.tangents[fullIdx * ModelImporterBase::TANGENT_ELEMENTS + 3u] = 1.0f;
        }

        // Bitangents:
        if (meshNormals && meshTangents)
        {
            const auto normal{ meshNormals[vIdx] };
            const auto tangent{ meshTangents[vIdx] };
            const auto bitangent{
                Vector3D::crossProduct(
                    Vector3D{ static_cast<float>(normal.x), static_cast<float>(normal.y), static_cast<float>(normal.z) },
                    Vector3D{ static_cast<float>(tangent.x), static_cast<float>(tangent.y), static_cast<float>(tangent.z) }
                )
            };
            pack.bitangents[fullIdx * ModelImporterBase::BITANGENT_ELEMENTS + 0u] = bitangent.x;
            pack.bitangents[fullIdx * ModelImporterBase::BITANGENT_ELEMENTS + 1u] = bitangent.y;
            pack.bitangents[fullIdx * ModelImporterBase::BITANGENT_ELEMENTS + 2u] = bitangent.z;
        }

        // Colors:
        if (meshColors)
        {
            const auto color{ meshColors[vIdx] };
            pack.colors[fullIdx * ModelImporterBase::COLOR_ELEMENTS + 0u] = static_cast<float>(color.x);
            pack.colors[fullIdx * ModelImporterBase::COLOR_ELEMENTS + 1u] = static_cast<float>(color.y);
            pack.colors[fullIdx * ModelImporterBase::COLOR_ELEMENTS + 2u] = static_cast<float>(color.z);
        }
    }
    pack.vertexCount = originalVertexCount + vertexCount;

    // Copy index data:
    pack.indices.resize(originalIndexCount + indexCount);
    for (std::size_t iIdx = 0u; iIdx < indexCount; ++iIdx)
    {
        const auto fullIdx{ iIdx + originalIndexCount };
        const auto &index{
            meshIndices[iIdx] < 0 ?
                static_cast<ModelImporterBase::IndexElementT>(-meshIndices[iIdx] - 1) :
                static_cast<ModelImporterBase::IndexElementT>(meshIndices[iIdx])
        };
        pack.indices[fullIdx] = index;
    }
    pack.indexCount = originalIndexCount + indexCount;
}

//#define DUMP_FBX

#ifdef DUMP_FBX

template <std::size_t MaxSize = 256u>
std::string dataViewToString(const ofbx::DataView &data)
{ char dataStr[MaxSize]{ }; data.toString(dataStr); return dataStr; }

template <typename T>
std::string arrayPropertyToString(const ofbx::IElementProperty &property)
{
    std::vector<T> values{ };
    const auto size{ property.getCount() };
    values.resize(std::min(size, 16));
    property.getValues(values.data(), property.getCount());

    std::stringstream ss{ };
    for (const T &v : values)
    { ss << v << " "; }

    if (size > property.getCount())
    { ss << "... and " << property.getCount() - size << " more"; }

    return ss.str();
}

std::string propertyToString(const ofbx::IElementProperty &property)
{
    std::stringstream ss{ };

    switch (property.getType())
    {
        case ofbx::IElementProperty::LONG:
        { ss << "Long: " << property.getValue().toU64(); break; }
        case ofbx::IElementProperty::FLOAT:
        { ss << "Float: " << property.getValue().toFloat(); break; }
        case ofbx::IElementProperty::DOUBLE:
        { ss << "Double: " << property.getValue().toDouble(); break; }
        case ofbx::IElementProperty::INTEGER:
        { ss << "Integer: " << property.getValue().toInt(); break; }
        case ofbx::IElementProperty::ARRAY_FLOAT:
        { ss << "Float array: " << arrayPropertyToString<float>(property); break; }
        case ofbx::IElementProperty::ARRAY_DOUBLE:
        { ss << "Double array: " << arrayPropertyToString<double>(property); break; }
        case ofbx::IElementProperty::ARRAY_INT:
        { ss << "Integer array: " << arrayPropertyToString<int>(property); break; }
        case ofbx::IElementProperty::ARRAY_LONG:
        { ss << "Long array: " << arrayPropertyToString<ofbx::u64>(property); break; }
        case ofbx::IElementProperty::STRING:
        { ss << "String: " << dataViewToString(property.getValue()); break; }
        default:
        { ss << "Other: type " << static_cast<char>(property.getType()); break; }
    }

    return ss.str();
}

void printElementInfo(const ofbx::IElement &parent, std::size_t depthLimit,
    std::ostream &out, std::size_t currentDepth = 0u, const std::string &indent = "")
{
    for (auto *element = parent.getFirstChild(); element; element = element->getSibling())
    {
        out << indent << "Element ID \"" << dataViewToString(element->getID()) << "\" : " << std::endl;
        for (auto *property = element->getFirstProperty(); property; property = property->getNext())
        { out << indent << "  Property: " << propertyToString(*property) << std::endl; }

        if (element->getFirstChild() && currentDepth < depthLimit)
        { printElementInfo(*element, depthLimit, out, currentDepth + 1u, indent + "  "); }
    }
}

void printNodeInfo(const ofbx::Object *node, std::size_t depthLimit,
    std::ostream &out, std::size_t currentDepth = 0u, const std::string &indent = "")
{
    for (std::size_t iii = 0u; node->resolveObjectLink(iii); ++iii)
    {
        const auto *child{ node->resolveObjectLink(iii) };

        std::stringstream label{ };
        char strOut[256]{ };
        switch (child->getType())
        {
            case ofbx::Object::Type::GEOMETRY: label << "geometry"; break;
            case ofbx::Object::Type::MESH: label << "mesh"; break;
            case ofbx::Object::Type::MATERIAL: label << "material"; break;
            case ofbx::Object::Type::ROOT: label << "root"; break;
            case ofbx::Object::Type::TEXTURE: label << "texture"; break;
            case ofbx::Object::Type::NULL_NODE: label << "null"; break;
            case ofbx::Object::Type::LIMB_NODE:
            {
                child->element.getID().toString(strOut);
                label << "limb node: " << strOut;
                break;
            }
            case ofbx::Object::Type::NODE_ATTRIBUTE:
            {
                const auto &attribute{ *dynamic_cast<const ofbx::NodeAttribute*>(child) };
                attribute.getAttributeType().toString(strOut);
                label << "node attribute: " << strOut;
                break;
            }
            case ofbx::Object::Type::CLUSTER: label << "cluster"; break;
            case ofbx::Object::Type::SKIN: label << "skin"; break;
            case ofbx::Object::Type::ANIMATION_STACK: label << "animation stack"; break;
            case ofbx::Object::Type::ANIMATION_LAYER: label << "animation layer"; break;
            case ofbx::Object::Type::ANIMATION_CURVE: label << "animation curve"; break;
            case ofbx::Object::Type::ANIMATION_CURVE_NODE: label << "animation curve node"; break;
            default: assert(false); break;
        }

        out << indent << "Node #" << iii << " | Label \"" << label.str() << "\" | Name \"" << child->name << "\" : " << std::endl;
        printElementInfo(child->element, depthLimit, out, currentDepth, indent + "  ");

        if (currentDepth < depthLimit)
        { out << indent << "  Children:\n"; printNodeInfo(child, depthLimit, out, currentDepth + 1u, indent + "\t"); }
    }
}

#endif // DUMP_FBX

std::string objectName(const ofbx::Object &object)
{ return object.name; }

bool objectNameEquals(const ofbx::Object &object, const std::string &name)
{ return objectName(object) == name; }

const ofbx::Object *findSkeletonRoot(const ofbx::IScene &scene)
{
    // The root node can be identified by its type (Null) and name "Root".
    const auto *root{ scene.getRoot() };
    const auto *object{ root };
    for (std::size_t iii = 0u; (object = root->resolveObjectLink(iii)); ++iii)
    { // Find the corresponding node.
        if (object->getType() == ofbx::Object::Type::NULL_NODE && objectNameEquals(*object, "Root"))
        { return object; }
    }

    return nullptr;
}

float getBoneLength(const ofbx::Object &node)
{ return static_cast<float>(node.getBoneLength(1.0)); }

float getBoneSize(const ofbx::Object &node)
{ return static_cast<float>(node.getBoneSize(1.0)); }

Vector3D getBoneRelativeOffset(const ofbx::Object &node)
{
    const auto offset{ node.getBonePosition({ 0.0, 0.0, 0.0 }) };
    return { static_cast<float>(offset.x), static_cast<float>(offset.y), static_cast<float>(offset.z) };
}

std::vector<const ofbx::Object*> getBoneChildren(const ofbx::Object &node, float &size)
{
    std::vector<const ofbx::Object*> result{ };

    const auto *object{ &node };
    for (std::size_t iii = 0u; (object = node.resolveObjectLink(iii)); ++iii)
    { // Find the corresponding node.
        if (object->getType() == ofbx::Object::Type::LIMB_NODE)
        { result.push_back(object); }
        else if (object->getType() == ofbx::Object::Type::NODE_ATTRIBUTE)
        { size = static_cast<float>(object->getBoneSize(-1.0)); }
    }

    return result;
}

TreeNodeData getBoneNodeData(
    const ofbx::Object &parent, const ofbx::Object &child,
    const Vector3D &parentAbsolutePosition, float currentSize)
{
    static constexpr auto BONE_WIDTH_MULTIPLIER{ 0.0006f };

    TreeNodeData result{ };

    const auto childAbsolutePosition{ parentAbsolutePosition + getBoneRelativeOffset(child) };
    const auto parentToChildLength{ getBoneLength(child) };
    const auto parentToChildOffset{ (childAbsolutePosition - parentAbsolutePosition) * parentToChildLength };

    result.pos = parentAbsolutePosition + parentToChildOffset;
    result.thickness = BONE_WIDTH_MULTIPLIER * currentSize;
    result.freezeThickness = true;

    return result;
}

void loadSkeleton(const ofbx::IScene &scene, ArrayTree &skeleton)
{
#ifdef DUMP_FBX
    auto out{ std::ofstream("fbx_dump.txt") };
    out << "Root: " << std::endl;
    //printNodeInfo(scene.getRoot()->resolveObjectLink(0u), 3u);
    printNodeInfo(scene.getRoot(), 10u, out);
    out << "Animation stack: " << std::endl;
    for (std::size_t iii = 0u; iii < scene.getAnimationStackCount(); ++iii)
    { printNodeInfo(scene.getAnimationStack(iii), 3u, out); }
    out.close();
#endif // DUMP_FBX

    /// @brief Record containing information about a single bone.
    struct BoneRecord
    {
        /// Starting node of the bone.
        const ofbx::Object *parent{ };
        /// ID of the starting node.
        ArrayTree::NodeIdT parentNodeId{ };
        /// Ending node of the bone.
        const ofbx::Object *child{ };
        /// Size of the bone.
        float currentSize{ };
    }; // struct BoneRecord

    // Get data for the root node.
    const auto rootNode{ findSkeletonRoot(scene) };
    float currentSize{ 0.0f };
    const auto rootChildBones{ getBoneChildren(*rootNode, currentSize) };
    if (currentSize <= 0.0f && !rootChildBones.empty())
    { getBoneChildren(*rootChildBones[0], currentSize); }

    if (!rootNode)
    { std::cerr << "Failed to find skeleton root node, ending!" << std::endl; return; }
    const auto rootNodeData{ getBoneNodeData(
        *rootNode, *rootNode,
        { 0.0f, 0.0f, 0.0f }, currentSize)
    };

    // Initialize the tree skeleton.
    skeleton.clearNodes();
    const auto rootNodeId{ skeleton.addRoot(rootNodeData) };

    // Initialize algorithm with all first order bones.
    std::stack<BoneRecord> bones{ };
    for (const auto *child : rootChildBones)
    { bones.push(BoneRecord{ rootNode, rootNodeId, child, currentSize }); }

    // Parse bones.
    while (!bones.empty())
    {
        // Move to the next bone.
        const auto boneRecord{ bones.top() }; bones.pop();

        // Get children and the current size.
        const auto childBones{ getBoneChildren(*boneRecord.child, currentSize) };

        // In case of no valid width, use the parents width.
        if (currentSize < 0.0f || currentSize > boneRecord.currentSize)
        { currentSize = boneRecord.currentSize; }

        // Calculate attributes of the current bone.
        const auto parentPosition{ skeleton.getNode(boneRecord.parentNodeId).data().pos };
        const auto nodeData{ getBoneNodeData(
            *boneRecord.parent, *boneRecord.child,
            parentPosition, currentSize)
        };

        // Add the bone.
        const auto nodeId{ skeleton.addNodeChild(boneRecord.parentNodeId, nodeData) };

        // Add all child bones.
        for (const auto *child : childBones)
        { bones.push(BoneRecord{ boneRecord.child, nodeId, child, currentSize }); }
    }
}

} // namespace impl

FbxImporter::FbxImporter()
{ clear(); }
FbxImporter::~FbxImporter()
{ /* Automatic */ }

void FbxImporter::clear()
{
    ModelImporterBase::clear();
    mSkeleton = { };
}

bool FbxImporter::import(const std::string &filePath)
{ return import(filePath, false); }

bool FbxImporter::import(const std::string &filePath, bool importSkeleton)
{
    // Clear past data.
    clear();

    // Open input file:
    std::ifstream inputFile(filePath, std::ios::binary | std::ios::in);
    if (!inputFile.is_open())
    { std::cerr << "Failed to open file: " << filePath << std::endl; return false; }

    // Load all data:
    std::vector<uint8_t> fbxData{ };
    inputFile.seekg(0, std::ios::end);
    fbxData.reserve(inputFile.tellg());
    inputFile.seekg(0, std::ios::beg);
    fbxData.assign(
        (std::istreambuf_iterator<char>(inputFile)),
        std::istreambuf_iterator<char>());

    // Load the input scene:
    // TODO - Protected destructor? Unable to delete -> memory leak?
    //const auto loadFlags{ static_cast<ofbx::u64>(ofbx::LoadFlags::TRIANGULATE) };
    const auto loadFlags{ static_cast<ofbx::u64>(0u) };
    DebugScopeStart
        std::cout << "Loading file: " << filePath << " using openfbx..." << std::endl;
    DebugScopeEnd
    const auto scenePtr{ ofbx::load(fbxData.data(), fbxData.size(), loadFlags) };
    if (!scenePtr)
    { std::cerr << "Failed to load FBX from: " << filePath << std::endl; return false; }
    const auto &scene{ *scenePtr };

    // Load mesh data:
    DebugScopeStart
        std::cout << "Loading mesh data..." << std::endl;
    DebugScopeEnd
    impl::loadMeshes(scene, {
        mVertices, mTextureCoordinates,
        mNormals, mTangets, mBitangents,
        mColors, mIndices,
        mVertexCount, mIndexCount
    });

    // Load skeleton data:
    if (importSkeleton)
    {
        DebugScopeStart
            std::cout << "loading skeleton data..." << std::endl;
        DebugScopeEnd
        impl::loadSkeleton(scene, mSkeleton);
    }

    DebugScopeStart
        std::cout << "Done" << std::endl;
    DebugScopeEnd

    return true;
}

const ArrayTree &FbxImporter::skeleton() const
{ return mSkeleton; }
ArrayTree &&FbxImporter::moveSkeleton()
{ return std::move(mSkeleton); }
bool FbxImporter::skeletonLoaded() const
{ return mSkeleton.nodeCount() > 0u; }

} // namespace treeio
