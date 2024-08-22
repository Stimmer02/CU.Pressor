#pragma once

/// @brief Interface for an observer that can be notified by a MultiNotifier
/// @tparam ID_TYPE type of the id representing the notifier
/// @tparam MESSAGE_TYPE type of the message that will be sent to the observer
template<typename ID_TYPE, typename MESSAGE_TYPE>
class IMultiObserver{
protected:
    friend class MultiNotifier<ID_TYPE, MESSAGE_TYPE>;

    /// @brief Notifies the class implementing this interface
    /// @param senderId id that helps to identify the sender
    /// @param message message that observer is notified about
    virtual void notify(const ID_TYPE& senderId, const MESSAGE_TYPE& message) = 0;
};