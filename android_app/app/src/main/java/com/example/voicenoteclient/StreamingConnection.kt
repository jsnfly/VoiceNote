import android.util.Log
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.*
import io.ktor.http.cio.websocket.*
import kotlinx.coroutines.channels.getOrElse

class ConnectionClosedException(message: String = "Connection is closed") : Exception(message)

class StreamingConnection(
    private val session: DefaultWebSocketSession,
    private val externalScope: CoroutineScope
) {
    private var receivedChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)
    private var sendChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)
    private var closed = false
    private var resettingRecv = false
    private var resettingSend = false

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
                        if (status == "RESET") {
                            reset(propagate = false)
                        } else if (resettingRecv) {
                            if (status == "INITIALIZING") {
                                resettingRecv = false
                                receivedChannel.send(message.data)
                            }
                        } else {
                            receivedChannel.send(message.data)
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
        } else if (resettingSend) {
            if (data["status"] == "INITIALIZING") {
                resettingSend = false
            } else {
                throw Exception("Reset in progress")
            }
        }

        externalScope.launch {
            sendChannel.send(data)
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

    fun reset(propagate: Boolean) {
        if (propagate) {
            resettingSend = false
            send(mapOf("status" to "RESET"))
        }
        resettingRecv = true
        resettingSend = true

        // Clear queues.
        runBlocking {
            while (!receivedChannel.isEmpty) { receivedChannel.receive() }
            while (!sendChannel.isEmpty) { sendChannel.receive() }
        }
    }

    suspend fun close() {
        session.close()
        receivedChannel.close()
        sendChannel.close()
    }
}