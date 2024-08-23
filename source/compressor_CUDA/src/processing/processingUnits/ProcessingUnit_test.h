#pragma once

#include "AProcessingUnit.h"

#include <string>

class ProcessingUnit_test : public AProcessingUnit {
public:
    ProcessingUnit_test(const std::string& name);

    void process() override;
    void activate();
    void deactivate();

    const std::string name;

private:
    virtual inline bool activationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const override;
    virtual inline bool deactivationFunction(const bool& active, const int& activeDependencies, const int& dependenciesSize) const override;
};