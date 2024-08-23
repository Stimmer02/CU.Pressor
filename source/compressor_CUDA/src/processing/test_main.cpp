#include "ProcessingQueue.h"
#include "processingUnits/ProcessingUnit_test.h"
#include "processingUnits/SoftDependencyGroup.h"

void test1();
void test2();
void test3();
void test4();

int main(){
    test4();

    return 0;
}

void test1(){
    ProcessingQueue queue;
    ProcessingUnit_test unit1("unit1");
    
    std::printf("\nTEST 1.1\n");
    queue.appendQueue(&unit1);
    queue.execute();

    std::printf("\nTEST 1.2\n");
    unit1.deactivate();
    queue.execute();

    std::printf("\nTEST 1.3\n");
    unit1.activate();
    queue.execute();
}

void test2(){
    ProcessingQueue queue;
    ProcessingUnit_test unit1("unit1");
    ProcessingUnit_test unit2("unit2");
    ProcessingUnit_test unit3("unit3");

    std::printf("\nTEST 2.1\n");
    queue.appendQueue(&unit1);
    queue.appendQueue(&unit2);
    queue.appendQueue(&unit3);
    queue.execute();

    std::printf("\nTEST 2.2\n");
    unit1.deactivate();
    queue.execute();

    std::printf("\nTEST 2.3\n");
    unit2.deactivate();
    unit3.deactivate();
    queue.execute();

    std::printf("\nTEST 2.4\n");
    unit1.activate();
    queue.execute();

    std::printf("\nTEST 2.5\n");
    unit2.activate();
    unit3.activate();
    unit1.deactivate();
    queue.execute();

    std::printf("\nTEST 2.6\n");
    unit2.deactivate();
    queue.execute();
}

void test3(){
    ProcessingQueue queue;
    ProcessingUnit_test unit1("unit1");
    ProcessingUnit_test unit2("unit2");
    ProcessingUnit_test unit3("unit3");
    ProcessingUnit_test unit4("unit4");

    queue.appendQueue(&unit1);
    queue.appendQueue(&unit2);
    queue.appendQueue(&unit3);
    queue.appendQueue(&unit4);
    unit4.deactivate();

    // simple dependency
    std::printf("\nTEST 3.1\n");
    unit1.registerDependency(&unit2);
    queue.execute();

    // broken simple dependency
    std::printf("\nTEST 3.2\n");
    unit2.deactivate();
    queue.execute();

    // restored simple dependency
    std::printf("\nTEST 3.3\n");
    unit2.activate();
    queue.execute();

    // double dependency
    std::printf("\nTEST 3.4\n");
    unit1.registerDependency(&unit3);
    queue.execute();

    // broken double dependency
    std::printf("\nTEST 3.5\n");
    unit3.deactivate();
    queue.execute();

    // broken double dependency 2
    std::printf("\nTEST 3.6\n");
    unit2.deactivate();
    queue.execute();

    // partially restored double dependency
    std::printf("\nTEST 3.7\n");
    unit3.activate();
    queue.execute();

    // restored double dependency
    std::printf("\nTEST 3.8\n");
    unit2.activate();
    queue.execute();

    // dependency chain
    std::printf("\nTEST 3.9\n");
    unit4.activate();
    unit2.registerDependency(&unit4);
    queue.execute();

    // broken dependency chain
    std::printf("\nTEST 3.10\n");
    unit4.deactivate();
    queue.execute();

    // broken dependency chain 2
    std::printf("\nTEST 3.11\n");
    unit2.deactivate();
    queue.execute();

    // partially restored dependency chain
    std::printf("\nTEST 3.12\n");
    unit4.activate();
    queue.execute();

    // restored dependency chain
    std::printf("\nTEST 3.13\n");
    unit2.activate();
    queue.execute();
}

void test4(){
    ProcessingQueue queue;
    ProcessingUnit_test unit1("unit1");
    ProcessingUnit_test unit2("unit2");
    ProcessingUnit_test unit3("unit3");
    ProcessingUnit_test unit4("unit4");
    ProcessingUnit_test unit5("unit5");
    ProcessingUnit_test unit6("unit6");

    SoftDependencyGroup group1;
    SoftDependencyGroup group2;

    queue.appendQueue(&unit1);
    queue.appendQueue(&unit2);
    queue.appendQueue(&unit3);
    queue.appendQueue(&unit4);   
    queue.appendQueue(&unit5);
    queue.appendQueue(&unit6);
    unit6.deactivate();

    // simple group
    std::printf("\nTEST 4.1\n");
    group1.registerUnit(&unit2);
    group1.registerUnit(&unit3);
    group1.registerUnit(&unit4);
    unit1.registerDependency(&group1);
    queue.execute();

    // not fully activated group 1
    std::printf("\nTEST 4.2\n");
    unit2.deactivate();
    queue.execute();

    // not fully activated group 2
    std::printf("\nTEST 4.3\n");
    unit3.deactivate();
    queue.execute();

    // fully deactivated group
    std::printf("\nTEST 4.4\n");
    unit4.deactivate();
    queue.execute();

    // recursively dependent
    std::printf("\nTEST 4.5\n");
    unit1.deactivate();
    unit2.activate();
    unit3.activate();
    unit4.activate();
    unit6.activate();
    group2.registerUnit(&unit6);
    group2.registerUnit(&group1);
    unit5.registerDependency(&group2);
    queue.execute();

    // broken group dependency
    std::printf("\nTEST 4.6\n");
    unit2.deactivate();
    unit3.deactivate();
    unit4.deactivate();
    queue.execute();

    // restored group dependency
    std::printf("\nTEST 4.7\n");
    unit2.activate();
    queue.execute();
}