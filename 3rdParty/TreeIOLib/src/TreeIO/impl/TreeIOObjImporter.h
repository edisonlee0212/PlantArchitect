/**
 * @author Tomas Polasek
 * @date 27.3.2020
 * @version 1.0
 * @brief Importer for OBJ meshes.
 */

#ifndef TREEIO_OBJ_IMPORTER_H
#define TREEIO_OBJ_IMPORTER_H

#include "TreeIOModelImporter.h"

namespace treeio
{

/// @brief OBJ file importer.
class ObjImporter : public ModelImporterBase
{
public:
    /// @brief Prepare importer for OBJ file importing.
    ObjImporter();
    /// @brief Clear all used memory.
    ~ObjImporter();

    /// @brief Clear all of the data.
    virtual void clear() override final;

    /// @brief Import OBJ file from given path and load its structure.
    virtual bool import(const std::string &filePath) override final;

    /// @brief Export OBJ file to given path.
    virtual bool exportTo(const std::string &filePath) override final;
private:
protected:
}; // class ObjImporter

} // namespace treeio

#endif // TREEIO_OBJ_IMPORTER_H
