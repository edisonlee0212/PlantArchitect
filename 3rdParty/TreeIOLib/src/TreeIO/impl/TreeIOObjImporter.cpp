/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Importer for OBJ meshes.
 */

#include "TreeIOObjImporter.h"

#include <algorithm>
#include <iostream>
#include <fstream>

#include "model_loader/ModelLoaderObj.h"

namespace treeio
{

ObjImporter::ObjImporter()
{ clear(); }
ObjImporter::~ObjImporter()
{ /* Automatic */ }

void ObjImporter::clear()
{ ModelImporterBase::clear(); }

bool ObjImporter::import(const std::string &filePath)
{
    // Clear past data.
    clear();

    // Import the OBJ file.
    ModelOBJ modelLoader{ };
    const auto importResult{ modelLoader.import(filePath.c_str(), false) };

    // Check for errors.
    if (!importResult)
    { std::cerr << "Object loader for \"" << filePath << "\" resulted in errors/warnings!" << std::endl; return false; }

    // Pre-allocate arrays:
    const auto vertexCount{ static_cast<std::size_t>(std::max(0, modelLoader.getNumberOfVertices())) };
    const auto indexCount{ static_cast<std::size_t>(std::max(0, modelLoader.getNumberOfIndices())) };
    mVertices.reserve(vertexCount * COORDINATE_ELEMENTS);
    mTextureCoordinates.reserve(vertexCount * TEX_COORD_ELEMENTS);
    mNormals.reserve(vertexCount * NORMAL_ELEMENTS);
    mTangets.reserve(vertexCount * TANGENT_ELEMENTS);
    mBitangents.reserve(vertexCount * BITANGENT_ELEMENTS);
    mColors.reserve(vertexCount * COLOR_ELEMENTS);
    mIndices.reserve(indexCount * 1u);

    // Copy imported vertex data:
    for (std::size_t iii = 0u; iii < vertexCount; ++iii)
    { // Process vertex data:
        const auto vertex{ modelLoader.getVertex(static_cast<int>(iii)) };

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
        mColors.push_back(1.0f);
        mColors.push_back(0.0f);
        mColors.push_back(1.0f);
        mColors.push_back(1.0f);
    }
    mVertexCount = vertexCount;

    // Store indices:
    for (std::size_t iii = 0u; iii < indexCount; ++iii)
    { mIndices.push_back(static_cast<IndexElementT>(modelLoader.getIndexBuffer()[iii])); }
    mIndexCount = indexCount;

    return true;
}

bool ObjImporter::exportTo(const std::string &filePath)
{
    std::ofstream out(filePath, std::ios::out);
    if (!out.is_open())
    { return false; }

    for (std::size_t iii = 1u; iii <= mVertexCount; ++iii)
    {
        out << "v " << mVertices[iii * COORDINATE_ELEMENTS + 0u] << " " <<
                       mVertices[iii * COORDINATE_ELEMENTS + 1u] << " " <<
                       mVertices[iii * COORDINATE_ELEMENTS + 2u] << "\n";
    }

    for (std::size_t iii = 1u; iii <= mVertexCount; ++iii)
    {
        out << "vn " << mNormals[iii * NORMAL_ELEMENTS + 0u] << " " <<
                        mNormals[iii * NORMAL_ELEMENTS + 1u] << " " <<
                        mNormals[iii * NORMAL_ELEMENTS + 2u] << "\n";
    }

    for (std::size_t iii = 1u; iii <= mVertexCount; ++iii)
    {
        out << "vt " << mTextureCoordinates[iii * TEX_COORD_ELEMENTS + 0u] << " " <<
                        mTextureCoordinates[iii * TEX_COORD_ELEMENTS + 1u] << " " <<
                        "0.0 \n";
    }

    for (std::size_t iii = 1u; iii < mIndexCount; iii += 3u)
    {
        out << "f " << iii + 0u << "/" << iii + 0u << "/" << iii + 0u << " " <<
                       iii + 1u << "/" << iii + 1u << "/" << iii + 1u << " " <<
                       iii + 2u << "/" << iii + 2u << "/" << iii + 2u << "\n";
    }

    out.close();

    return true;
}

} // namespace treeio
