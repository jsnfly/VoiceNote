package com.example.voicenoteclient

import android.util.Log
import io.ktor.client.*
import io.ktor.client.features.websocket.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.selects.select

class WebSocketManager(
    private val host: String,
    private val port: Int,
    private val scope: CoroutineScope,
    private val onConnectionStatusChanged: (ConnectionStatus) -> Unit = {}
) {
    private val client = HttpClient {
        install(WebSockets)
    }
    private var connection: StreamingConnection? = null
    private val stopSignal = Channel<Unit>(1)
    private var connectionJob: Job? = null
    private var closed = false

    init {
        setupWebSocketConnection()
    }

    fun currentConnectionOrNull(): StreamingConnection? = connection

    private fun setupWebSocketConnection() {
        connectionJob = scope.launch {
            while (isActive && !closed) {
                onConnectionStatusChanged(ConnectionStatus.CONNECTING)
                try {
                    client.webSocket(host = host, port = port) {
                        val activeConnection = StreamingConnection(this, this@launch)
                        connection = activeConnection
                        activeConnection.run()
                        onConnectionStatusChanged(ConnectionStatus.CONNECTED)

                        val stopWaiter = async { stopSignal.receive() }
                        val closeWaiter = async { activeConnection.awaitClosed() }
                        select<Unit> {
                            stopWaiter.onAwait {}
                            closeWaiter.onAwait {}
                        }
                        stopWaiter.cancel()
                        closeWaiter.cancel()
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e(AppLog.TAG, "WebSocket connection failed for $host:$port", e)
                    onConnectionStatusChanged(ConnectionStatus.DISCONNECTED)
                    delay(RECONNECT_DELAY_MS)
                } finally {
                    connection = null
                    if (!closed) {
                        onConnectionStatusChanged(ConnectionStatus.DISCONNECTED)
                    }
                }
            }
        }
    }

    fun closeConnection() {
        closed = true
        connection?.close()
        stopSignal.trySend(Unit)
        connectionJob?.cancel()
        client.close()
        onConnectionStatusChanged(ConnectionStatus.DISCONNECTED)
    }

    private companion object {
        const val RECONNECT_DELAY_MS = 1_500L
    }
}
