#pragma once

#include "../MultiNotifier.h"

#include <vector>

/// @brief Base class for processing units managed by ProcessingQueue.
/// Processing units logic should be implemented outside of this class.
class AProcessingUnit : public IMultiObserver<int, bool>{
friend class SoftDependencyGroup;
public:
    AProcessingUnit() : active(true), activeDependencies(0), hardDeactivation(false){}

    /// @brief Executes the processing unit logic
    virtual void process() = 0;

    /// @brief Registers a observer to be notified when the state of this processing unit changes
    /// @param observer IMultiObserver to register
    /// @param id identifier sent to observer when notified apart from the message
    virtual void registerObserver(IMultiObserver<int, bool>* observer, int id);

    /// @brief Registers a dependency. By default all dependencies have to be active for the processing unit to be active
    /// @param dependency AProcessingUnit to register as a dependency
    void registerDependency(AProcessingUnit* dependency);

    /// @brief Returns the state of the processing unit
    /// @return true if the unit is active
    bool isActive() const;
    
protected:
    MultiNotifier<int, bool> notifier;

    /// @brief Sets the hard deactivation flag. Both dependencies and the hard deactivation flag have to allow the unit to be active
    /// @param active if false then the unit will be deactivated until the flag is set to true
    void setActive(bool active);

    /// @brief Called by notify method on every dependency activation to determine if the processing unit should be activated
    /// @param active unit state
    /// @param activeDependencies number of active dependencies 
    /// @param dependenciesSize number of all dependencies
    /// @return true if the unit should be activated
    virtual bool activationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const;

    /// @brief Called by notify method on every dependency deactivation to determine if the processing unit should be deactivated
    /// @param active unit state
    /// @param activeDependencies number of active dependencies 
    /// @param dependenciesSize number of all dependencies
    /// @return true if the unit should be deactivated 
    virtual bool deactivationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const;

private:
    std::vector<AProcessingUnit*> dependencies;
    int activeDependencies;
    bool active;
    bool hardDeactivation;

    // method called by dependency to notify about state change
    // it also notifies observers if dependency state change changes state of this processing unit
    virtual void notify(const int& senderId, const bool& message) override;
};