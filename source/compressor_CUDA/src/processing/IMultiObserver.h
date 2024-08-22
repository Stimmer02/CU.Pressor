#pragma once

template<typename ID_TYPE, typename MESSAGE_TYPE>
class IMultiObserver{
protected:
    friend class MultiNotifier<ID_TYPE, MESSAGE_TYPE>;
    virtual void notify(const ID_TYPE& senderId, const MESSAGE_TYPE& message) = 0;
};