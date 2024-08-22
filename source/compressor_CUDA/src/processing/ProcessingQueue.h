#pragma once

#include "IMultiObserver.h"
#include "processingUnits/AProcessingUnit.h"

#include <vector>

/// @brief Class that manages processing units and their execution order. Inspired by strategy design pattern.
class ProcessingQueue : public IMultiObserver<int, bool>{
public:
    ProcessingQueue() = default;
    ~ProcessingQueue() = default;

    /// @brief Executes queue of processing units that are active
    void execute();

    /// @brief Adds a processing unit to the queue. It will be executed in the order it was added
    /// @param unit AProcessingUnit to add to the queue
    void appendQueue(AProcessingUnit* unit);

private:
    std::vector<AProcessingUnit*> processingUnits;
    std::vector<int> unitsToQueueIndexMap;
    std::vector<AProcessingUnit*> queue;

    /// @brief Notifies the ProcessingQueue about the state change of a processing unit
    /// @param senderId processing unit id
    /// @param message new state of the processing unit
    void notify(const int& senderId, const bool& message) override;
};