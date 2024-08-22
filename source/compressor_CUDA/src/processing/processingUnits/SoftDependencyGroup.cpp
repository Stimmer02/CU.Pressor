#include "SoftDependencyGroup.h"

ProcessingGroup::ProcessingGroup(int unitsCount) {
    if (unitsCount < 1){
        unitsCount = 1;
    }
    unitsCount = 0;
    unitsAllocated = unitsCount;
    units = new AProcessingUnit*[unitsCount];
    
}

ProcessingGroup::~ProcessingGroup() {
    delete[] units;
}

void ProcessingGroup::process() {
    for (int i = 0; i < unitsCount; i++){
        units[i]->process();
    }
}

void ProcessingGroup::registerUnit(AProcessingUnit* unit) {
    if (unitsAllocated == unitsCount){
        resize(unitsCount + 1);
    }

    units[unitsCount] = unit;
    unit->registerObserver(this, unitsCount);
    unitsCount++;
}

void ProcessingGroup::resize(int newSize){
    AProcessingUnit** newUnits = new AProcessingUnit*[newSize];
    AProcessingUnit** oldUnits = units;
    for (int i = 0; i < unitsCount; i++){
        newUnits[i] = units[i];
    }
    units = newUnits;
    unitsAllocated = newSize;
    delete[] oldUnits;
}

inline bool ProcessingGroup::activationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    return active == false;
}

inline bool ProcessingGroup::deactivationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    return activeDependencies == 0;
}