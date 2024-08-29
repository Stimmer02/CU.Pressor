#pragma once

#include "../MultiNotifier.h"

#include <vector>

/// @brief Base class for processing units managed by ProcessingQueue.
/// Processing units logic should be implemented outside of this class.
class AProcessingUnit : public IMultiObserver<int, bool>{
friend class SoftDependencyGroup;
public:
    /// @brief Constructor
    AProcessingUnit();

    /// @brief Executes the processing unit logic
    virtual void process() = 0;

    /// @brief Registers a observer to be notified when the state of this processing unit changes
    /// @param observer IMultiObserver to register
    /// @param id identifier sent to observer when notified apart from the message
    virtual void registerObserver(IMultiObserver<int, bool>* observer, int id);

    /// @brief Registers a dependency. By default all dependencies have to be active for the processing unit to be active
    /// @param dependency AProcessingUnit to register as a dependency
    void registerDependency(AProcessingUnit* dependency);

    /// @brief Registers an exclusion. If any exclusion is active then the processing unit will be deactivated
    /// @param exclusion AProcessingUnit to register as an exclusion
    void registerExclusion(AProcessingUnit* exclusion);

    /// @brief Returns the state of the processing unit
    /// @return true if the unit is active
    bool isActive() const;
    
protected:
    MultiNotifier<int, bool> notifier;

    /// @brief Sets the hard deactivation flag. Both dependencies and the hard deactivation flag have to allow the unit to be active
    /// @param active if false then the unit will be deactivated until the flag is set to true
    void setActive(bool active);

    /// @brief Determines if the processing unit should active or inactive
    /// @param activeDependencies number of active dependencies 
    /// @param dependenciesSize number of all dependencies
    /// @param activeExclusions number of active exclusions
    /// @param exclusionsSize number of all exclusions
    /// @return true if the unit should be deactivated 
    virtual bool activationFunction(const int& activeDependencies, const int& dependenciesSize, const int& activeExclusions, const int& exclusionsSize) const;

    /// @brief checks current state and updates it if necessary informing observers
    void checkAndUpdateState();

private:
    int registeredExclusions;
    int activeExclusions;

    int registeredDependencies;
    int activeDependencies;

    bool active;
    bool hardDeactivation;

    // method called by dependency to notify about state change
    // it also notifies observers if dependency state change changes state of this processing unit
    virtual void notify(const int& senderId, const bool& message) override;

};