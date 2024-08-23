#include "SoftDependencyGroup.h"

SoftDependencyGroup::SoftDependencyGroup(int preallocatedSize) : AProcessingUnit(){
    if (preallocatedSize < 1){
        preallocatedSize = 1;
    }
    unitsCount = 0;
    unitsAllocated = preallocatedSize;
    units = new AProcessingUnit*[preallocatedSize];
    
}

SoftDependencyGroup::~SoftDependencyGroup() {
    delete[] units;
}

void SoftDependencyGroup::process() {
    for (int i = 0; i < unitsCount; i++){
        if (units[i]->isActive()){
            units[i]->process();
        }
    }
}

void SoftDependencyGroup::registerUnit(AProcessingUnit* unit) {
    if (unitsAllocated == unitsCount){
        resize(unitsCount + 1);
    }

    units[unitsCount] = unit;
    unitsCount++;
    registerDependency(unit);
}

void SoftDependencyGroup::resize(int newSize){
    AProcessingUnit** newUnits = new AProcessingUnit*[newSize];
    AProcessingUnit** oldUnits = units;
    for (int i = 0; i < unitsCount; i++){
        newUnits[i] = units[i];
    }
    units = newUnits;
    unitsAllocated = newSize;
    delete[] oldUnits;
}

inline bool SoftDependencyGroup::activationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    return active == false;
}

inline bool SoftDependencyGroup::deactivationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    return activeDependencies == 0;
}