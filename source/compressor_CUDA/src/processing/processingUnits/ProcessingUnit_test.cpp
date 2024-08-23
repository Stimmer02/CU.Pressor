#include "ProcessingUnit_test.h"

ProcessingUnit_test::ProcessingUnit_test(const std::string& name) : AProcessingUnit(), name(name){}

void ProcessingUnit_test::process(){
    std::printf("%s processed\n", name.c_str());
}

void ProcessingUnit_test::activate(){
    setActive(true);
}

void ProcessingUnit_test::deactivate(){
    if(isActive()){
        std::printf("%s deactivated\n", name.c_str());
    }
    setActive(false);
}

inline bool ProcessingUnit_test::activationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    bool result = activeDependencies == dependenciesSize;
    if (result){
        std::printf("%s activated\n", name.c_str());
    }
    return result;
}

inline bool ProcessingUnit_test::deactivationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const{
    bool result = active;
    if (result){
        std::printf("%s deactivated\n", name.c_str());
    }
    return result;
}