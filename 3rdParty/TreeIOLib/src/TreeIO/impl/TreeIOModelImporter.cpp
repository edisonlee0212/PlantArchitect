/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Base importer class.
 */

#include "TreeIOModelImporter.h"

#include <iostream>

namespace treeio
{

ModelImporterBase::ModelImporterBase()
{ clear(); }
ModelImporterBase::~ModelImporterBase()
{ /* Automatic */ }

void ModelImporterBase::clear()
{
    mVertices = { };
    mTextureCoordinates = { };
    mNormals = { };
    mTangets = { };
    mBitangents = { };
    mColors = { };

    mIndices = { };

    mVertexCount = { };
    mIndexCount = { };
}

bool ModelImporterBase::exportTo(const std::string &filePath)
{ return false; }

bool ModelImporterBase::importVertices(const std::vector<Vertex> &vertices)
{
    clear();
    if (vertices.size() % 3u != 0u)
    { return false; }

    // Pre-allocate arrays:
    const auto vertexCount{ vertices.size() };
    const auto indexCount{ vertices.size()};
    mVertices.reserve(vertexCount * COORDINATE_ELEMENTS);
    mTextureCoordinates.reserve(vertexCount * TEX_COORD_ELEMENTS);
    mNormals.reserve(vertexCount * NORMAL_ELEMENTS);
    mTangets.reserve(vertexCount * TANGENT_ELEMENTS);
    mBitangents.reserve(vertexCount * BITANGENT_ELEMENTS);
    mColors.reserve(vertexCount * COLOR_ELEMENTS);
    mIndices.reserve(indexCount * 1u);

    // Copy vertex data:
    for (const auto &vertex : vertices)
    { // Process vertex data:
        // Position:
        mVertices.push_back(vertex.position[0]);
        mVertices.push_back(vertex.position[1]);
        mVertices.push_back(vertex.position[2]);

        // Texture coordinates:
        mTextureCoordinates.push_back(vertex.texCoord[0]);
        mTextureCoordinates.push_back(vertex.texCoord[1]);

        // Normal:
        mNormals.push_back(vertex.normal[0]);
        mNormals.push_back(vertex.normal[1]);
        mNormals.push_back(vertex.normal[2]);

        // Tangent:
        mTangets.push_back(vertex.tangent[0]);
        mTangets.push_back(vertex.tangent[1]);
        mTangets.push_back(vertex.tangent[2]);
        mTangets.push_back(vertex.tangent[3]);

        // Bitangent:
        mBitangents.push_back(vertex.bitangent[0]);
        mBitangents.push_back(vertex.bitangent[1]);
        mBitangents.push_back(vertex.bitangent[2]);

        // TODO - Color:
        mColors.push_back(vertex.color[0]);
        mColors.push_back(vertex.color[1]);
        mColors.push_back(vertex.color[2]);
        mColors.push_back(vertex.color[3]);
    }
    mVertexCount = vertexCount;

    // Store indices:
    for (std::size_t iii = 0u; iii < indexCount; ++iii)
    { mIndices.push_back(static_cast<IndexElementT>(iii)); }
    mIndexCount = indexCount;

    return true;
}

const std::vector<ModelImporterBase::VertexElementT> &ModelImporterBase::positions() const
{ return mVertices; }
std::vector<ModelImporterBase::VertexElementT> &&ModelImporterBase::movePositions()
{ return std::move(mVertices); }
std::size_t ModelImporterBase::positionCount() const
{ return mVertices.size() / COORDINATE_ELEMENTS; }

const std::vector<ModelImporterBase::VertexElementT> &ModelImporterBase::texCoordinates() const
{ return mTextureCoordinates; }
std::vector<ModelImporterBase::VertexElementT> &&ModelImporterBase::moveTexCoordinates()
{ return std::move(mTextureCoordinates); }
std::size_t ModelImporterBase::texCoordinateCount() const
{ return mTextureCoordinates.size() / TEX_COORD_ELEMENTS; }

const std::vector<ModelImporterBase::VertexElementT> &ModelImporterBase::normals() const
{ return mNormals; }
std::vector<ModelImporterBase::VertexElementT> &&ModelImporterBase::moveNormals()
{ return std::move(mNormals); }
std::size_t ModelImporterBase::normalCount() const
{ return mNormals.size() / NORMAL_ELEMENTS; }

const std::vector<ModelImporterBase::VertexElementT> &ModelImporterBase::tangents() const
{ return mTangets; }
std::vector<ModelImporterBase::VertexElementT> &&ModelImporterBase::moveTangents()
{ return std::move(mTangets); }
std::size_t ModelImporterBase::tangentCount() const
{ return mTangets.size() / TANGENT_ELEMENTS; }

const std::vector<ModelImporterBase::VertexElementT> &ModelImporterBase::bitangents() const
{ return mBitangents; }
std::vector<ModelImporterBase::VertexElementT> &&ModelImporterBase::moveBitangents()
{ return std::move(mBitangents); }
std::size_t ModelImporterBase::bitangentCount() const
{ return mBitangents.size() / BITANGENT_ELEMENTS; }

const std::vector<ModelImporterBase::VertexElementT> &ModelImporterBase::colors() const
{ return mColors; }
std::vector<ModelImporterBase::VertexElementT> &&ModelImporterBase::moveColors()
{ return std::move(mColors); }
std::size_t ModelImporterBase::colorCount() const
{ return mColors.size() / COLOR_ELEMENTS; }

std::size_t ModelImporterBase::vertexCount() const
{ return mVertexCount; }
ModelImporterBase::Vertex ModelImporterBase::getVertex(std::size_t idx) const
{
    Vertex result{ };

    if (idx < positionCount())
    {
        result.position[0u] = positions()[idx * COORDINATE_ELEMENTS + 0u];
        result.position[1u] = positions()[idx * COORDINATE_ELEMENTS + 1u];
        result.position[2u] = positions()[idx * COORDINATE_ELEMENTS + 2u];
    }

    if (idx < texCoordinateCount())
    {
        result.texCoord[0u] = texCoordinates()[idx * TEX_COORD_ELEMENTS + 0u];
        result.texCoord[1u] = texCoordinates()[idx * TEX_COORD_ELEMENTS + 1u];
    }

    if (idx < normalCount())
    {
        result.normal[0u] = normals()[idx * NORMAL_ELEMENTS + 0u];
        result.normal[1u] = normals()[idx * NORMAL_ELEMENTS + 1u];
        result.normal[2u] = normals()[idx * NORMAL_ELEMENTS + 2u];
    }

    if (idx < tangentCount())
    {
        result.tangent[0u] = tangents()[idx * TANGENT_ELEMENTS + 0u];
        result.tangent[1u] = tangents()[idx * TANGENT_ELEMENTS + 1u];
        result.tangent[2u] = tangents()[idx * TANGENT_ELEMENTS + 2u];
        result.tangent[3u] = tangents()[idx * TANGENT_ELEMENTS + 3u];
    }

    if (idx < bitangentCount())
    {
        result.bitangent[0u] = bitangents()[idx * BITANGENT_ELEMENTS + 0u];
        result.bitangent[1u] = bitangents()[idx * BITANGENT_ELEMENTS + 1u];
        result.bitangent[2u] = bitangents()[idx * BITANGENT_ELEMENTS + 2u];
    }

    return result;
}

const std::vector<ModelImporterBase::IndexElementT> &ModelImporterBase::indices() const
{ return mIndices; }
std::vector<ModelImporterBase::IndexElementT> &&ModelImporterBase::moveIndices()
{ return std::move(mIndices); }
std::size_t ModelImporterBase::indexCount() const
{ return mIndexCount; }

} // namespace treeio
