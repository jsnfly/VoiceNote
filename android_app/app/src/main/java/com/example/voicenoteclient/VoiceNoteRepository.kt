package com.example.voicenoteclient

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.*

class VoiceNoteRepository(
    private val context: Context,
    private val externalScope: CoroutineScope
) {
    private val appContext = context.applicationContext
    private val preferences = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val _connectionStatus = MutableStateFlow(ConnectionStatus.CONNECTING)
    private var webSocketManager = createWebSocketManager(loadServerSettings())

    val messageChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)
    val connectionStatus = _connectionStatus.asStateFlow()
    val serverSettings: ServerSettings
        get() = loadServerSettings()

    private var receiveJob: Job? = null
    private var audioRecord: AudioRecord? = null
    private var audioPlayer: AudioPlayer? = null

    private val sampleRate = AppDefaults.INPUT_SAMPLE_RATE
    private val audioConfig = mapOf(
        ProtocolKey.FORMAT to 8,
        ProtocolKey.CHANNELS to 1,
        ProtocolKey.RATE to sampleRate
    )
    private var communicationId: String? = null

    fun initialize() {
        if (receiveJob != null) return
        receiveJob = externalScope.launch(Dispatchers.IO) {
            recvDataFromSocket()
        }
    }

    private suspend fun recvDataFromSocket() {
        while (currentCoroutineContext().isActive) {
            try {
                val connection = webSocketManager.currentConnectionOrNull()
                if (connection == null) {
                    delay(SOCKET_POLL_DELAY_MS)
                    continue
                }

                val messages = connection.receive()
                for (msg in messages) {
                    messageChannel.send(msg)
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: ConnectionClosedException) {
                delay(SOCKET_POLL_DELAY_MS)
            } catch (e: Exception) {
                Log.e(AppLog.TAG, "Failed to receive WebSocket data", e)
                delay(SOCKET_POLL_DELAY_MS)
            }
            delay(SOCKET_POLL_DELAY_MS)
        }
    }

    fun sendAction(action: String, savePath: String?) {
        externalScope.launch(Dispatchers.IO) {
            val id = UUID.randomUUID().toString()
            communicationId = id
            val connection = awaitCurrentConnection()
            connection.reset(id)
            connection.send(
                mapOf(
                    ProtocolKey.ACTION to action,
                    ProtocolKey.ID to id,
                    ProtocolKey.SAVE_PATH to savePath,
                    ProtocolKey.STATUS to ProtocolStatus.INITIALIZING
                )
            )
        }
    }

    suspend fun startRecording(): Boolean {
        val recorder = audioRecord ?: run {
            Log.w(AppLog.TAG, "Cannot start recording before AudioRecord is initialized")
            return false
        }

        val id = UUID.randomUUID().toString()
        communicationId = id
        val connection = awaitCurrentConnection()
        connection.reset(id)
        connection.send(
            mapOf(
                ProtocolKey.AUDIO_CONFIG to audioConfig,
                ProtocolKey.ID to id,
                ProtocolKey.STATUS to ProtocolStatus.INITIALIZING
            )
        )
        recorder.startRecording()
        return true
    }

    fun stopRecording() {
        val recorder = audioRecord ?: return
        val id = communicationId ?: return

        runCatching { recorder.stop() }
            .onFailure { Log.w(AppLog.TAG, "AudioRecord stop failed", it) }

        var numOutBytes = writeAudioDataToSocket()
        while (numOutBytes > 0) {
            numOutBytes = writeAudioDataToSocket()
        }
        webSocketManager.awaitedConnectionOrNull()?.send(
            mapOf(
                ProtocolKey.STATUS to ProtocolStatus.FINISHED,
                ProtocolKey.AUDIO to ByteArray(0),
                ProtocolKey.ID to id
            )
        )
    }

    fun writeAudioDataToSocket(): Int {
        val id = communicationId ?: return 0
        val recorder = audioRecord ?: return 0
        val outBuffer = ByteArray(recorder.bufferSizeInFrames * 2)
        val numOutBytes = recorder.read(outBuffer, 0, outBuffer.size)
        if (numOutBytes > 0) {
            webSocketManager.awaitedConnectionOrNull()?.send(
                mapOf(
                    ProtocolKey.AUDIO to outBuffer.sliceArray(0 until numOutBytes),
                    ProtocolKey.ID to id,
                    ProtocolKey.STATUS to ProtocolStatus.RECORDING
                )
            )
        }
        return numOutBytes
    }

    fun playAudio(audioData: ByteArray) {
        if (audioPlayer == null) {
            audioPlayer = AudioPlayer(externalScope)
        }
        audioPlayer?.playAudio(audioData.toFloatArray())
    }

    fun stopPlayback() {
        audioPlayer?.terminateAudioProcessing()
        audioPlayer = null
    }

    fun setupAudioRecord(activity: AppCompatActivity) {
        audioRecord = com.example.voicenoteclient.setupAudioRecord(
            activity, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
    }

    fun updateServerSettings(settings: ServerSettings) {
        val normalizedSettings = settings.copy(host = settings.host.trim())
        if (normalizedSettings == serverSettings) return

        preferences.edit()
            .putString(PREF_HOST, normalizedSettings.host)
            .putInt(PREF_PORT, normalizedSettings.port)
            .apply()

        webSocketManager.closeConnection()
        webSocketManager = createWebSocketManager(normalizedSettings)
    }

    fun reconnect() {
        webSocketManager.closeConnection()
        webSocketManager = createWebSocketManager(serverSettings)
    }

    fun close() {
        stopPlayback()
        runCatching { audioRecord?.stop() }
        audioRecord?.release()
        audioRecord = null
        receiveJob?.cancel()
        webSocketManager.closeConnection()
        messageChannel.close()
    }

    private fun loadServerSettings(): ServerSettings {
        return ServerSettings(
            host = preferences.getString(PREF_HOST, AppDefaults.HOST) ?: AppDefaults.HOST,
            port = preferences.getInt(PREF_PORT, AppDefaults.PORT)
        )
    }

    private fun createWebSocketManager(settings: ServerSettings): WebSocketManager {
        return WebSocketManager(settings.host, settings.port, externalScope) { status ->
            _connectionStatus.value = status
        }
    }

    private fun WebSocketManager.awaitedConnectionOrNull(): StreamingConnection? {
        return currentConnectionOrNull().also {
            if (it == null) {
                Log.w(AppLog.TAG, "No WebSocket connection available")
            }
        }
    }

    private suspend fun awaitCurrentConnection(): StreamingConnection {
        while (currentCoroutineContext().isActive) {
            webSocketManager.currentConnectionOrNull()?.let { return it }
            delay(SOCKET_POLL_DELAY_MS)
        }
        throw CancellationException("Stopped waiting for WebSocket connection")
    }

    private companion object {
        const val PREFS_NAME = "voice_note_settings"
        const val PREF_HOST = "host"
        const val PREF_PORT = "port"
        const val SOCKET_POLL_DELAY_MS = 50L
    }
}
