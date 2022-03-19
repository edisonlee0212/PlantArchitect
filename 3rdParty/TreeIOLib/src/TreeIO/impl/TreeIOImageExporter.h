/**
 * @author Tomas Polasek
 * @date 14.5.2020
 * @version 1.0
 * @brief Exporter for various image formats.
 */

#ifndef TREEIO_IMAGE_EXPORTER_H
#define TREEIO_IMAGE_EXPORTER_H

#include <iterator>
#include <vector>

#include "TreeIOImageImporter.h"

namespace treeio
{

// Forward declaration.
namespace impl
{ class ImageExporterImpl; }

/// @brief Exporter for various image formats.
class ImageExporter
{
public:
    /// @brief Type used to store color information.
    using ChannelFormatT = unsigned char;

    /// @brief Initialize the internal structure.
    ImageExporter();
    /// @brief Cleanup and destroy
    ~ImageExporter();

    /// @brief Clear all of the data.
    void clear();

    /// @brief Load image from given image importer and prepare it for export.
    void loadImage(const ImageImporter &importer);

    /// @brief Load image from given array / pointer and prepare it for export.
    template <typename ArrT>
    void loadImage(const ArrT &pixels, std::size_t width, std::size_t height, bool normalize = false);
    /// @brief Load image from given span and prepare it for export.
    template <typename ItT>
    void loadImage(const ItT &begin, const ItT &end, std::size_t width, std::size_t height, bool normalize = false);

    /// @brief Load image from given pointer and prepare it for export.
    void loadImage(const std::vector<ChannelFormatT> &pixels, std::size_t width, std::size_t height);

    /// @brief Export image file to given path.
    bool exportImage(const std::string &filePath) const;

    /// @brief get width of the currently loaded image.
    std::size_t width() const;
    /// @brief Get height of the currently loaded image.
    std::size_t height() const;

    /// @brief Conversion operator for ease of access.
    operator const ChannelFormatT*() const;

    /// @brief Access currently loaded data.
    const ChannelFormatT *data() const;
private:
    /// Internal implementation of the exporter.
    std::shared_ptr<impl::ImageExporterImpl> mImpl{ };
protected:
}; // class ImageExporter

} // namespace treeio

// Template implementation begin.

namespace treeio
{

template <typename ArrT>
void ImageExporter::loadImage(const ArrT &pixels, std::size_t width, std::size_t height, bool normalize)
{ using std::begin; using std::end; loadImage(begin(pixels), end(pixels), width, height, normalize); }

template <typename ItT>
void ImageExporter::loadImage(const ItT &begin, const ItT &end, std::size_t width, std::size_t height, bool normalize)
{
    std::vector<ChannelFormatT> convertedData{ };

    if (normalize)
    {
        const auto normalizedData{ treeutil::convertImageNormalizedRGB(begin, end) };
        convertedData.resize(normalizedData.size() * 3u);
        for (std::size_t iii = 0u; iii < normalizedData.size(); ++iii)
        {
            const auto quantized{ normalizedData[iii] * std::numeric_limits<ChannelFormatT>::max() };
            convertedData[iii * 3u + 0u] = static_cast<ChannelFormatT>(quantized.x);
            convertedData[iii * 3u + 1u] = static_cast<ChannelFormatT>(quantized.y);
            convertedData[iii * 3u + 2u] = static_cast<ChannelFormatT>(quantized.z);
        }
    }
    else
    {
        const auto itDistance{ std::distance(begin, end) };
        if (itDistance < 0)
        { return; }
        const auto elementCount{ static_cast<std::size_t>(itDistance) };
        convertedData.resize(elementCount);

        auto it{ begin };
        for (std::size_t iii = 0u; iii < elementCount && it != end; ++iii, ++it)
        { convertedData[iii] = static_cast<ChannelFormatT>(*it); }
    }

    loadImage(std::move(convertedData), width, height);
}

} // namespace treeio

// Template implementation end.

#endif // TREEIO_IMAGE_EXPORTER_H
