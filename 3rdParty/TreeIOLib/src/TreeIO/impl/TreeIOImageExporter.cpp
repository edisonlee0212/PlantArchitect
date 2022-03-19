/**
 * @author Tomas Polasek
 * @date 14.5.2020
 * @version 1.0
 * @brief Exporter for various image formats.
 */

#include "TreeIOImageExporter.h"

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

/// @brief Internal implementation of the ImageExporter.
class ImageExporterImpl
{
public:
    /// Internal image exporter implementation.
    cimg_library::CImg<ImageExporter::ChannelFormatT> imageExporter{ };
private:
protected:
}; // class ImageExporterImpl

}

ImageExporter::ImageExporter()
{ clear(); }
ImageExporter::~ImageExporter()
{ /* Automatic */}

void ImageExporter::clear()
{ mImpl = std::make_shared<impl::ImageExporterImpl>(); }

void ImageExporter::loadImage(const ImageImporter &importer)
{ loadImage(importer.begin(), importer.end(), importer.width(), importer.height()); }

void ImageExporter::loadImage(const std::vector<ChannelFormatT> &pixels,
    std::size_t width, std::size_t height)
{
    // Clear past data.
    clear();

    const auto channels{ pixels.size() / (width * height)};
    if (channels == 0)
    { return; }

    mImpl->imageExporter = decltype(mImpl->imageExporter)(width, height, 1u, channels, 0u);
    cimg_forXYC(mImpl->imageExporter, xxx, yyy, ccc)
    { mImpl->imageExporter(xxx, yyy, ccc) = pixels[(xxx + width * (height - 1 - yyy)) * channels + ccc]; }
}

bool ImageExporter::exportImage(const std::string &filePath) const
{
    try
    { // Save the image.
        mImpl->imageExporter.save(filePath.c_str());
    } catch (cimg_library::CImgException&)
    { // The saving failed.
        return false;
    }

    return true;
}

std::size_t ImageExporter::width() const
{ return mImpl->imageExporter.width(); }
std::size_t ImageExporter::height() const
{ return mImpl->imageExporter.height(); }

ImageExporter::operator const ChannelFormatT*() const
{ return data(); }
const ImageExporter::ChannelFormatT *ImageExporter::data() const
{ return mImpl->imageExporter.data(); }

} // namespace treeio
