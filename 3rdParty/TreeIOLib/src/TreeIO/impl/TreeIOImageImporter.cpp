/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Importer for various image formats.
 */

#include "TreeIOImageImporter.h"

#include <iostream>

// We use CImg only for loading. Disable image display capabilities.
#define cimg_display 0
#define cimg_use_jpeg
#define cimg_use_png
#include "cimg/CImg.h"
#undef cimg_display

namespace treeio
{

namespace impl
{

/// @brief Internal implementation of the ImageImporter.
class ImageImporterImpl
{
public:
    /// Internal image loader implementation.
    cimg_library::CImg<ImageImporter::ChannelFormatT> imageLoader{ };
private:
protected:
}; // class ImageImporterImpl

/// @brief Get maximum color channel value for given type.
template <typename T>
static constexpr T MaxColor();
template <>
constexpr unsigned char MaxColor()
{ return 255u; }
template <>
constexpr float MaxColor()
{ return 1.0f; }

/// @brief Get minimum color channel value for given type.
template <typename T>
static constexpr T MinColor();
template <>
constexpr unsigned char MinColor()
{ return 0u; }
template <>
constexpr float MinColor()
{ return 0.0f; }

}

ImageImporter::ImageImporter()
{ clear(); }
ImageImporter::~ImageImporter()
{ /* Automatic */}

void ImageImporter::clear()
{ mImpl = std::make_shared<impl::ImageImporterImpl>(); }

ImageImporter ImageImporter::createDebugPattern(std::size_t width, std::size_t height, std::size_t tileSize)
{
    static constexpr auto COLOR_CHANNELS{ 3u };
    static constexpr ImageImporter::ChannelFormatT TILE_COLOR[COLOR_CHANNELS]{
        impl::MaxColor<ImageImporter::ChannelFormatT>(),
        impl::MinColor<ImageImporter::ChannelFormatT>(),
        impl::MaxColor<ImageImporter::ChannelFormatT>()
    };
    static constexpr ImageImporter::ChannelFormatT BG_COLOR[COLOR_CHANNELS]{
        impl::MinColor<ImageImporter::ChannelFormatT>(),
        impl::MinColor<ImageImporter::ChannelFormatT>(),
        impl::MinColor<ImageImporter::ChannelFormatT>()
    };
    static constexpr auto DEPTH{ 1u };
    static constexpr auto DEFAULT_VALUE{ 0u };
    const auto pixelCount{ width * height * DEPTH };

    ImageImporter result{ };
    result.mImpl->imageLoader = cimg_library::CImg<ImageImporter::ChannelFormatT >(
        width, height, DEPTH, COLOR_CHANNELS, DEFAULT_VALUE);
    auto &image{ result.mImpl->imageLoader };
    for (std::size_t iii = 0u; iii < pixelCount; ++iii)
    {
        const auto mx{ iii % width };
        const auto my{ iii / height };
        const auto tx{ mx / tileSize};
        const auto ty{ my / tileSize};

        if ((tx + ty) % 2u)
        { // Within a tile.
            image[iii + pixelCount * 0u] = TILE_COLOR[0u];
            image[iii + pixelCount * 1u] = TILE_COLOR[1u];
            image[iii + pixelCount * 2u] = TILE_COLOR[2u];
        }
        else
        { // Outside of a tile.
            image[iii + pixelCount * 0u] = BG_COLOR[0u];
            image[iii + pixelCount * 1u] = BG_COLOR[1u];
            image[iii + pixelCount * 2u] = BG_COLOR[2u];
        }
    }

    return result;
}

bool ImageImporter::importImage(const std::string &filePath)
{
    // Clear past data.
    clear();

    try
    { // Load the image.
        mImpl->imageLoader.load(filePath.c_str());
    } catch (cimg_library::CImgException&)
    { // The loading failed -> Generate a dummy image to avoid crashes.
        const auto debugPattern{ createDebugPattern(32u, 32u, 4u) };
        mImpl = debugPattern.mImpl;
        return false;
    }

    return true;
}

void ImageImporter::permuteAxes(const std::string &permutation)
{ mImpl->imageLoader.permute_axes(permutation.c_str()); }

std::size_t ImageImporter::width() const
{ return mImpl->imageLoader.width(); }
std::size_t ImageImporter::height() const
{ return mImpl->imageLoader.height(); }

ImageImporter::operator const ChannelFormatT*() const
{ return data(); }
const ImageImporter::ChannelFormatT *ImageImporter::data() const
{ return mImpl->imageLoader.data(); }

const ImageImporter::ChannelFormatT *ImageImporter::begin() const
{ return mImpl->imageLoader.data(); }

const ImageImporter::ChannelFormatT *ImageImporter::end() const
{ return mImpl->imageLoader.data() + mImpl->imageLoader.size(); }

ImageImporter::ChannelFormatT ImageImporter::pixelRawValue(std::size_t xPos, std::size_t yPos, std::size_t color) const
{ return mImpl->imageLoader(xPos, yPos, color); }

float ImageImporter::pixelValue(std::size_t xPos, std::size_t yPos, std::size_t color) const
{ return static_cast<float>(pixelRawValue(xPos, yPos, color)) / impl::MaxColor<ChannelFormatT>(); }

} // namespace treeio
