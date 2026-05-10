package com.example.voicenoteclient

import android.util.Log
import io.ktor.http.cio.websocket.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.channels.Channel

class ConnectionClosedException(message: String = "Connection is closed") : Exception(message)

class StreamingConnection(
    private val session: DefaultWebSocketSession,
    private val externalScope: CoroutineScope
) {
    private var receivedChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)
    private var sendChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)
    private val closeSignal = Channel<Unit>(1)
    private var closed = false
    var communicationID: String? = null

    fun run() {
        externalScope.launch {
            val receiveJob = launch { receiveToChannel() }
            val sendJob = launch { sendFromChannel() }
            joinAll(receiveJob, sendJob)
        }
    }

    private suspend fun receiveToChannel() {
        try {
            for (frame in session.incoming) {
                when (frame) {
                    is Frame.Text -> {
                        val message = Message.fromDataString(frame.readText())
                        val status = message.data[ProtocolKey.STATUS] as? String
                        val id = message.data[ProtocolKey.ID] as? String
                        if (id == null) {
                            Log.d(AppLog.TAG, "Discarding msg without id.")
                        } else if (status == ProtocolStatus.RESET) {
                            reset(id = id, propagate = false)
                        } else if (isValidMessage(id)) {
                            receivedChannel.send(message.data)
                        } else {
                            Log.d(AppLog.TAG, "Discarding msg with id $id.")
                        }
                    }
                    else -> {}
                }
            }
        } finally {
            receivedChannel.close()
            notifyClosed()
        }
    }

    private suspend fun sendFromChannel() {
        try {
            for (data in sendChannel) {
                Log.d(AppLog.TAG, "Sending $data")
                session.send(Message(data).encode())
            }
        } finally {
            sendChannel.close()
            notifyClosed()
        }
    }

    fun send(data: Map<String, Any?>) {
        val id = data[ProtocolKey.ID] as? String
        if (closed) {
            throw ConnectionClosedException("Connection is closed")
        } else if (id == null) {
            throw IllegalArgumentException("Message is missing id")
        } else if (isValidMessage(id)) {
            val result = sendChannel.trySend(data)
            if (result.isFailure) {
                throw ConnectionClosedException("Connection send queue is closed")
            }
        } else {
            throw Exception("Reset in progress")
        }
    }

    suspend fun receive(): List<Map<String, Any?>> {
        if (closed) throw ConnectionClosedException("Connection is closed")
        val received = mutableListOf<Map<String, Any?>>()

        // Non-blocking read of all currently available messages
        while (true) {
            val next = receivedChannel.tryReceive().getOrNull() ?: break
            received.add(next)
        }
        return received
    }

    suspend fun reset(id: String, propagate: Boolean = true) {
        communicationID = id

        // Clear queues non-blockingly
        while (true) {
            receivedChannel.tryReceive().getOrNull() ?: break
        }
        while (true) {
            sendChannel.tryReceive().getOrNull() ?: break
        }

        if (propagate) {
            send(mapOf(ProtocolKey.ID to id, ProtocolKey.STATUS to ProtocolStatus.RESET))
        }
    }

    private fun isValidMessage(id: String): Boolean {
        return communicationID == null || communicationID == id
    }

    suspend fun awaitClosed() {
        closeSignal.receive()
    }

    fun close() {
        closed = true
        notifyClosed()
        receivedChannel.close()
        sendChannel.close()
        externalScope.launch { session.close() }
    }

    private fun notifyClosed() {
        closeSignal.trySend(Unit)
    }
}
