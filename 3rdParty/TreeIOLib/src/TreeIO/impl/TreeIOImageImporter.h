/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Importer for various image formats.
 */

#ifndef TREEIO_IMAGE_IMPORTER_H
#define TREEIO_IMAGE_IMPORTER_H

#include <string>
#include <memory>

#include "TreeIOUtils.h"

namespace treeio
{

// Forward declaration.
namespace impl
{ class ImageImporterImpl; }

/// @brief Importer for various image formats.
class ImageImporter
{
public:
    /// @brief Type used to store color information.
    using ChannelFormatT = unsigned char;

    /// @brief Initialize the internal structure.
    ImageImporter();
    /// @brief Cleanup and destroy
    ~ImageImporter();

    /// @brief Clear all of the data.
    void clear();

    /// @brief Helper used to create importer filled with debug pattern image.
    static ImageImporter createDebugPattern(std::size_t width, std::size_t height, std::size_t tileSize);

    /// @brief Import image file from given path and load its structure.
    bool importImage(const std::string &filePath);

    /// @brief Permute axes of the currently loaded image.
    void permuteAxes(const std::string &permutation);

    /// @brief get width of the currently loaded image.
    std::size_t width() const;
    /// @brief Get height of the currently loaded image.
    std::size_t height() const;

    /// @brief Conversion operator for ease of access.
    operator const ChannelFormatT*() const;

    /// @brief Access the imported data.
    const ChannelFormatT *data() const;

    /// @brief Begin iterator for the imported data.
    const ChannelFormatT *begin() const;
    /// @brief End iterator for the imported data.
    const ChannelFormatT *end() const;

    /// @brief Get color value at given position.
    ChannelFormatT pixelRawValue(std::size_t xPos, std::size_t yPos, std::size_t color) const;

    /// @brief Get color value at given position.
    float pixelValue(std::size_t xPos, std::size_t yPos, std::size_t color) const;
private:
    /// Internal implementation of the importer.
    std::shared_ptr<impl::ImageImporterImpl> mImpl{ };
protected:
}; // class ImageImporter

} // namespace treeio

#endif // TREEIO_IMAGE_IMPORTER_H
