#pragma once

#include "AProcessingUnit.h"

/// @brief Allows to group multiple processing units and use it as a dependency
/// In order to activate the group at least one of the units has to be active
class ProcessingGroup : public AProcessingUnit{
public:
    /// @brief Creates a new ProcessingGroup
    /// @param unitsCount preallocated size of the group (resize is time expensive)
    ProcessingGroup(int unitsCount = 1);
    ~ProcessingGroup();

    /// @brief Executes process method on all units registered by registerUnit()
    void process() override;

    /// @brief Registers a new dependency and allows the unit to be invoked by the group's process method
    /// @param unit AProcessingUnit to register
    virtual void registerUnit(AProcessingUnit* unit);


private:
    AProcessingUnit** units;
    int unitsCount;
    int unitsAllocated;

    inline bool activationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const override;
    inline bool deactivationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const override;

    /// @brief resizes the units array
    /// @param newSize new size of the array
    void resize(int newSize);
};