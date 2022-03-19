/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Importer for FBX meshes and skeletons.
 */

#ifndef TREEIO_FBX_IMPORTER_H
#define TREEIO_FBX_IMPORTER_H

#include "TreeIOUtils.h"
#include "TreeIOTree.h"
#include "TreeIOModelImporter.h"

namespace treeio
{

/// @brief FBX file importer.
class FbxImporter : public ModelImporterBase
{
public:
    /// @brief Prepare importer for FBX file importing.
    FbxImporter();
    /// @brief Clear all used memory.
    ~FbxImporter();

    /// @brief Clear all of the data.
    virtual void clear() override final;

    /// @brief Import OBJ file from given path and load its structure.
    virtual bool import(const std::string &filePath) override final;

    /// @brief Import OBJ file from given path and load its structure.
    bool import(const std::string &filePath, bool importSkeleton);

    /// @brief Get currently loaded skeleton.
    const ArrayTree &skeleton() const;
    /// @brief Get currently loaded skeleton.
    ArrayTree &&moveSkeleton();
    /// @brief Did we also load a skeleton of the model?
    bool skeletonLoaded() const;
private:
    /// Tree containing the imported skeleton.
    ArrayTree mSkeleton{ };
protected:
}; // class FbxImporter

} // namespace treeio

#endif // TREEIO_FBX_IMPORTER_H
