/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Base model importer class.
 */

#ifndef TREEIO_MODEL_IMPORTER_H
#define TREEIO_MODEL_IMPORTER_H

#include <string>
#include <vector>

namespace treeio
{

/// @brief Base for all model importers.
class ModelImporterBase
{
public:
    /// @brief Type used for each vertex attribute.
    using VertexElementT = float;
    /// @brief Number of coordinates in a single position (XYZ).
    static constexpr auto COORDINATE_ELEMENTS{ 3u };
    /// @brief Number of coordinates in a single texture coordinate (XY).
    static constexpr auto TEX_COORD_ELEMENTS{ 2u };
    /// @brief Number of coordinates in a single normal (XYZ).
    static constexpr auto NORMAL_ELEMENTS{ 3u };
    /// @brief Number of coordinates in a single tangent (XYZ, handedness).
    static constexpr auto TANGENT_ELEMENTS{ 4u };
    /// @brief Number of coordinates in a single bitangent (XYZ).
    static constexpr auto BITANGENT_ELEMENTS{ 3u };
    /// @brief Number of elements in a single color (RGBA).
    static constexpr auto COLOR_ELEMENTS{ 4u };

    /// @brief Type used for each index.
    using IndexElementT = unsigned int;

    /// @brief Vertex representation used by this model.
    struct Vertex
    {
        /// Position in 3D space.
        VertexElementT position[COORDINATE_ELEMENTS]{ };
        /// Texture coordinate.
        VertexElementT texCoord[TEX_COORD_ELEMENTS]{ };
        /// Normal in 3D space.
        VertexElementT normal[NORMAL_ELEMENTS]{ };
        /// Tangent in 3D space and handedness.
        VertexElementT tangent[TANGENT_ELEMENTS]{ };
        /// Bitangent in 3D space.
        VertexElementT bitangent[BITANGENT_ELEMENTS]{ };
        /// Color channels.
        VertexElementT color[COLOR_ELEMENTS]{ };
    }; // struct Vertex

    /// @brief Prepare importer for OBJ file importing.
    ModelImporterBase();
    /// @brief Clear all used memory.
    ~ModelImporterBase();

    /// @brief Clear all of the data.
    virtual void clear();

    /// @brief Import file from given path and load its structure.
    virtual bool import(const std::string &filePath) = 0;

    /// @brief Export file to given path.
    virtual bool exportTo(const std::string &filePath);

    /// @brief Import given list of vertices. Each three consecutive vertices make up an triangle.
    bool importVertices(const std::vector<Vertex> &vertices);

    /// @brief Get list of currently loaded positions. Each position has 3 coordinates (XYZ).
    const std::vector<VertexElementT> &positions() const;
    /// @brief Get list of currently loaded positions. Each position has 3 coordinates (XYZ).
    std::vector<VertexElementT> &&movePositions();
    /// @brief Number of positions in the positions array.
    std::size_t positionCount() const;

    /// @brief Get list of currently loaded texture coordinates. Each texture coordinate has 2 elements (XY).
    const std::vector<VertexElementT> &texCoordinates() const;
    /// @brief Get list of currently loaded texture coordinates. Each texture coordinate has 2 elements (XY).
    std::vector<VertexElementT> &&moveTexCoordinates();
    /// @brief Number of texture coordinates in the texture coordinates array.
    std::size_t texCoordinateCount() const;

    /// @brief Get list of currently loaded normals. Each normal has 3 elements (XYZ).
    const std::vector<VertexElementT> &normals() const;
    /// @brief Get list of currently loaded normals. Each normal has 3 elements (XYZ).
    std::vector<VertexElementT> &&moveNormals();
    /// @brief Number of normals in the normals array.
    std::size_t normalCount() const;

    /// @brief Get list of currently loaded tangents. Each tangent has 4 elements (XYZ, handedness).
    const std::vector<VertexElementT> &tangents() const;
    /// @brief Get list of currently loaded tangents. Each tangent has 4 elements (XYZ, handedness).
    std::vector<VertexElementT> &&moveTangents();
    /// @brief Number of tangents in the tangents array.
    std::size_t tangentCount() const;

    /// @brief Get list of currently loaded bitangents. Each bitangent has 3 elements (XYZ).
    const std::vector<VertexElementT> &bitangents() const;
    /// @brief Get list of currently loaded bitangents. Each bitangent has 3 elements (XYZ).
    std::vector<VertexElementT> &&moveBitangents();
    /// @brief Number of bitangents in the bitangents array.
    std::size_t bitangentCount() const;

    /// @brief Get list of currently loaded colors. Each color has 4 elements (RGBA).
    const std::vector<VertexElementT> &colors() const;
    /// @brief Get list of currently loaded colors. Each color has 4 elements (RGBA).
    std::vector<VertexElementT> &&moveColors();
    /// @brief Number of colors in the colors array.
    std::size_t colorCount() const;

    /// @brief Get total number of vertices.
    std::size_t vertexCount() const;
    /// @brief Fill Vertex structure with data for vertex on given index. Unavailable values are left as defaults.
    Vertex getVertex(std::size_t idx) const;

    /// @brief Get list of currently loaded indices. Indices points to the vertices array.
    const std::vector<IndexElementT> &indices() const;
    /// @brief Get list of currently loaded indices. Indices points to the vertices array.
    std::vector<IndexElementT> &&moveIndices();
    /// @brief Number of indices in the indices array.
    std::size_t indexCount() const;
private:
protected:
    /// Vertex data. Each vertex has 3 coordinates (XYZ).
    std::vector<VertexElementT> mVertices{ };
    /// Texture coordinate data. Each texture coordinate has 2 coordinates (XY).
    std::vector<VertexElementT> mTextureCoordinates{ };
    /// Normal data. Each normal has 3 coordinates (XYZ).
    std::vector<VertexElementT> mNormals{ };
    /// Tangent data. Each tangent has 4 coordinates (XYZ, handedness).
    std::vector<VertexElementT> mTangets{ };
    /// Bitangent data. Each bitangent has 3 coordinates (XYZ).
    std::vector<VertexElementT> mBitangents{ };
    /// Color data. Each color has 4 elements (RGBA).
    std::vector<VertexElementT> mColors{ };

    /// Index data. Indices point to the vertices array.
    std::vector<IndexElementT> mIndices{ };

    /// Total number of vertices.
    std::size_t mVertexCount{ 0u };
    /// Total number of indices.
    std::size_t mIndexCount{ 0u };
}; // class ModelImporterBase

} // namespace treeio

#endif // TREEIO_MODEL_IMPORTER_H
