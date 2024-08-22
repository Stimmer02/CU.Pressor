#include "AProcessingUnit.h"

void AProcessingUnit::registerObserver(IMultiObserver<int, bool>* observer, int id){
    notifier.registerObserver(observer, id);
}

void AProcessingUnit::registerDependency(AProcessingUnit* dependency){
    dependency->registerObserver(this, dependencies.size());
    dependencies.push_back(dependency);
    if (dependency->active){
        activeDependencies++;
    }
}

bool AProcessingUnit::isActive() const{
    return active;
}

void AProcessingUnit::setActive(bool active){
    if (hardDeactivation == active){
        return;
    }
    hardDeactivation = active;
    if (active){
        if (activationFunction(active, activeDependencies, dependencies.size())){
            this->active = true;
            notifier.notifyObservers(true);
        }
    } else {
        if (deactivationFunction(active, activeDependencies, dependencies.size())){
            this->active = false;
            notifier.notifyObservers(false);
        }
    }
}

inline bool AProcessingUnit::activationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    return activeDependencies == dependenciesSize;
}

inline bool AProcessingUnit::deactivationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    return active;
}

void AProcessingUnit::notify(const int& senderId, const bool& message){
    if (message){
        activeDependencies++;
        if (hardDeactivation == false && activationFunction(active, activeDependencies, dependencies.size())){
            active = true;
            notifier.notifyObservers(true);
        }
    } else {
        activeDependencies--;
        if (deactivationFunction(active, activeDependencies, dependencies.size())){
            active = false;
            notifier.notifyObservers(false);
        }
    }
}