package com.example.voicenoteclient

import io.ktor.client.*
import io.ktor.client.features.websocket.*
import io.ktor.http.cio.websocket.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import StreamingConnection

class WebSocketManager(private val host: String, private val port: Int) {
    private val client = HttpClient {
        install(WebSockets)
    }
    lateinit var connection: StreamingConnection
    private val stopSignal = Channel<Boolean>(1)

    init {
        setupWebSocketConnection()
    }

    fun isConnectionInitialized(): Boolean = ::connection.isInitialized

    private fun setupWebSocketConnection() {
        // Using CoroutineScope tied to application lifecycle, or some long-lived scope
        GlobalScope.launch {
            client.webSocket(host = host, port = port) {
                connection = StreamingConnection(this, this@launch)
                connection.run()

                // Keep the WebSocket session open until explicitly told to close
                stopSignal.receive()
            }
        }
    }

    suspend fun closeConnection() {
        if (::connection.isInitialized) {
            connection.close()
        }
        stopSignal.send(true)
        client.close()
    }
}