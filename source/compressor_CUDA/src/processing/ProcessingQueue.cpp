#include "ProcessingQueue.h"

void ProcessingQueue::execute(){
    for (int i = 0; i < queue.size(); i++){
        queue[i]->process();
    }
}

void ProcessingQueue::appendQueue(AProcessingUnit* unit){
    unit->registerObserver(this, processingUnits.size());
    processingUnits.push_back(unit);
    if (unit->isActive()){
        unitsToQueueIndexMap.push_back(queue.size());
        queue.push_back(unit);
    } else {
        unitsToQueueIndexMap.push_back(-1);
    }
}

void ProcessingQueue::notify(const int& senderId, const bool& message){
    if (message){
        if (unitsToQueueIndexMap[senderId] != -1){
            throw std::runtime_error("ProcessingQueue::notify: Processing unit already in the queue");
        }
        int index = senderId - 1;
        for (; index >= 0; index--){
            if (unitsToQueueIndexMap[index] != -1){
                break;
            }
        }
        if (index == -1){
            queue.insert(queue.begin(), processingUnits[senderId]);
            unitsToQueueIndexMap[senderId] = 0;
        } else {
            queue.insert(queue.begin() + unitsToQueueIndexMap[index] + 1, processingUnits[senderId]);
            unitsToQueueIndexMap[senderId] = unitsToQueueIndexMap[index] + 1;
        }
        for (int i = senderId + 1; i < unitsToQueueIndexMap.size(); i++){
            if (unitsToQueueIndexMap[i] != -1){
                unitsToQueueIndexMap[i]++;
            }
        }
    } else {
        if (unitsToQueueIndexMap[senderId] == -1){
            throw std::runtime_error("ProcessingQueue::notify: Processing unit not found in the queue");
        }
        queue.erase(queue.begin() + unitsToQueueIndexMap[senderId]);
        unitsToQueueIndexMap[senderId] = -1;
        for (int i = senderId + 1; i < unitsToQueueIndexMap.size(); i++){
            if (unitsToQueueIndexMap[i] != -1){
                unitsToQueueIndexMap[i]--;
            }
        }
    }
}