import android.util.Log
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.*
import io.ktor.http.cio.websocket.*

class ConnectionClosedException(message: String = "Connection is closed") : Exception(message)

class StreamingConnection(
    private val session: DefaultWebSocketSession,
    private val externalScope: CoroutineScope
) {
    private var receivedChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)
    private var sendChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)
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
                        val status = message.data["status"] as? String
                        val id = message.data["id"] as String
                        if (status == "RESET") {
                            reset(id = id, propagate = false)
                        } else if (isValidMessage(id)) {
                            receivedChannel.send(message.data)
                        } else {
                            Log.d("XXXXX", "Discarding msg with id $id.")
                        }
                    }
                    else -> {}
                }
            }
        } finally {
            receivedChannel.close()
        }
    }

    private suspend fun sendFromChannel() {
        try {
            for (data in sendChannel) {
                Log.d("XXXXX", "Sending $data")
                session.send(Message(data).encode())
            }
        } finally {
            sendChannel.close()
        }
    }

    fun send(data: Map<String, Any?>) {
        if (closed) {
            throw ConnectionClosedException("Connection is closed")
        } else if (isValidMessage(data["id"] as String)) {
            externalScope.launch { sendChannel.send(data) }
        } else {
            throw Exception("Reset in progress")
        }
    }

    fun receive(): List<Map<String, Any?>> = runBlocking {
        if (closed) throw ConnectionClosedException("Connection is closed")
        val received = mutableListOf<Map<String, Any?>>()

        while (!receivedChannel.isEmpty) {
            received.add(receivedChannel.receive())
        }
        received
    }

    fun reset(id: String, propagate: Boolean = true) {
        communicationID = id

        // Clear queues.
        runBlocking {
            while (!receivedChannel.isEmpty) { receivedChannel.receive() }
            while (!sendChannel.isEmpty) { sendChannel.receive() }
        }

        if (propagate) {
            send(mapOf("id" to id, "status" to "RESET"))
        }
    }

    private fun isValidMessage(id: String): Boolean {
        return communicationID == null || communicationID == id
    }

    suspend fun close() {
        session.close()
        receivedChannel.close()
        sendChannel.close()
    }
}