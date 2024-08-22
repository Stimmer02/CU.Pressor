#pragma once

#include "IMultiObserver.h"

template<typename ID_TYPE, typename MESSAGE_TYPE>
class MultiNotifier{
public:
    MultiNotifier(const int observerCount = 1);
    ~MultiNotifier();
    void notifyObservers(const MESSAGE_TYPE& message);
    void registerObserver(IMultiObserver<ID_TYPE, MESSAGE_TYPE>* observer, ID_TYPE id);

private:
    IMultiObserver<ID_TYPE, MESSAGE_TYPE>** observers;
    ID_TYPE* obtainedIds;
    int count;
    int allocated;

    void resize(int newSize);
};



template<typename ID_TYPE, typename MESSAGE_TYPE>
MultiNotifier<ID_TYPE, MESSAGE_TYPE>::MultiNotifier(const int observerCount){
    count = 0;
    allocated = observerCount < 1 ? 1 : observerCount;
    observers = new IMultiObserver<ID_TYPE, MESSAGE_TYPE>*[observerCount];
    obtainedIds = new ID_TYPE[observerCount];
}

template<typename ID_TYPE, typename MESSAGE_TYPE>
MultiNotifier<ID_TYPE, MESSAGE_TYPE>::~MultiNotifier(){
    delete[] observers;
    delete[] obtainedIds;
}

template<typename ID_TYPE, typename MESSAGE_TYPE>
void MultiNotifier<ID_TYPE, MESSAGE_TYPE>::notifyObservers(const MESSAGE_TYPE& message){
    for(int i = 0; i < count; i++){
        observers[i]->notify(obtainedIds[i], message);
    }
}

template<typename ID_TYPE, typename MESSAGE_TYPE>
void MultiNotifier<ID_TYPE, MESSAGE_TYPE>::registerObserver(IMultiObserver<ID_TYPE, MESSAGE_TYPE>* observer, ID_TYPE id){
    if(count == allocated){
        resize(allocated + 1);
    }
    observers[count] = observer;
    obtainedIds[count] = id;
    count++;
}

template<typename ID_TYPE, typename MESSAGE_TYPE>
void MultiNotifier<ID_TYPE, MESSAGE_TYPE>::resize(int newSize){
    IMultiObserver<ID_TYPE, MESSAGE_TYPE>** newObservers = new IMultiObserver<ID_TYPE, MESSAGE_TYPE>*[newSize];
    ID_TYPE* newObtainedIds = new ID_TYPE[newSize];

    IMultiObserver<ID_TYPE, MESSAGE_TYPE>** oldObservers = observers;
    ID_TYPE* oldObtainedIds = obtainedIds;

    for (int i = 0; i < count; i++){
        newObservers[i] = observers[i];
        newObtainedIds[i] = obtainedIds[i];
    }

    observers = newObservers;
    obtainedIds = newObtainedIds;
    allocated = newSize;

    delete[] oldObservers;
    delete[] oldObtainedIds;
    
}