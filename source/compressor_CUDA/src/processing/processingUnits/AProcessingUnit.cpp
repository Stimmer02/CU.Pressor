#include "AProcessingUnit.h"

AProcessingUnit::AProcessingUnit() : active(true), registeredExclusions(0), activeExclusions(0), registeredDependencies(0), activeDependencies(0), hardDeactivation(false){}

void AProcessingUnit::registerObserver(IMultiObserver<int, bool>* observer, int id){
    notifier.registerObserver(observer, id);
}

void AProcessingUnit::registerDependency(AProcessingUnit* dependency){
    dependency->registerObserver(this, registeredDependencies);
    registeredDependencies++;
    if (dependency->isActive()){
        activeDependencies++;
    }
    checkAndUpdateState();
}

void AProcessingUnit::registerExclusion(AProcessingUnit* exclusion){
    exclusion->registerObserver(this, registeredExclusions | 0x80000000);
    registeredExclusions++;
    if (exclusion->isActive()){
        activeExclusions++;
    }
    checkAndUpdateState();
}

bool AProcessingUnit::isActive() const{
    return active;
}

void AProcessingUnit::setActive(bool active){
    if (hardDeactivation != active){
        return;
    }
    hardDeactivation = !active;
    checkAndUpdateState();
}

bool AProcessingUnit::activationFunction(const int& activeDependencies, const int& dependenciesSize, const int& activeExclusions, const int& exclusionsSize) const {
    return activeDependencies == dependenciesSize && activeExclusions == 0;
}

void AProcessingUnit::checkAndUpdateState(){
    bool correctState = hardDeactivation == false && activationFunction(activeDependencies, registeredDependencies, activeExclusions, registeredExclusions);

    if (active == correctState){
        return;
    }
    
    active = correctState;
    notifier.notifyObservers(active);
}


void AProcessingUnit::notify(const int& senderId, const bool& message){
    bool dependency = !(bool)(senderId & 0x80000000); // true if dependency, false if exclusion
    int& valueToUpdate = (dependency) ? activeDependencies : activeExclusions;
    valueToUpdate += (message) ? 1 : -1;
    checkAndUpdateState();
}