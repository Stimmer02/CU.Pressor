#pragma once

#include "IMultiObserver.h"
#include "processingUnits/AProcessingUnit.h"

#include <vector>

class ProcessingQueue : public IMultiObserver<int, bool>{
public:
    ProcessingQueue() = default;
    ~ProcessingQueue() = default;

    void execute();
    void appendQueue(AProcessingUnit* unit);

private:
    std::vector<AProcessingUnit*> processingUnits;
    std::vector<bool> processingUnitsStatus;
    std::vector<AProcessingUnit*> queue;

    void notify(const int& senderId, const bool& message) override;
};