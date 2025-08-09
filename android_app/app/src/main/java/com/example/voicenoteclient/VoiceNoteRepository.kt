package com.example.voicenoteclient

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.*

class VoiceNoteRepository(
    private val context: Context,
    private val externalScope: CoroutineScope
) {
    private val webSocketManager = WebSocketManager("192.168.0.154", 12345, externalScope)
    val messageChannel = Channel<Map<String, Any?>>(Channel.UNLIMITED)

    private lateinit var audioRecord: AudioRecord
    private var audioPlayer: AudioPlayer? = null

    private val sampleRate = 44100
    private val audioConfig = mapOf("format" to 8, "channels" to 1, "rate" to sampleRate)
    private var communicationId: String? = null

    fun initialize() {
        externalScope.launch {
            recvDataFromSocket()
        }
    }

    private suspend fun recvDataFromSocket() {
        while (true) {
            while (!webSocketManager.isConnectionInitialized()) {
                delay(50)
            }
            val messages = webSocketManager.connection.receive()
            for (msg in messages) {
                messageChannel.send(msg)
            }
            delay(50)
        }
    }

    fun sendAction(action: String, savePath: String?) {
        externalScope.launch {
            val id = UUID.randomUUID().toString()
            communicationId = id
            webSocketManager.connection.reset(id)
            webSocketManager.connection.send(mapOf(
                "action" to action,
                "id" to id,
                "save_path" to savePath,
                "status" to "INITIALIZING"
            ))
        }
    }

    fun startRecording(topic: String) {
        externalScope.launch {
            val id = UUID.randomUUID().toString()
            communicationId = id
            webSocketManager.connection.reset(id)
            webSocketManager.connection.send(mapOf(
                "audio_config" to audioConfig,
                "id" to id,
                "status" to "INITIALIZING",
                "topic" to topic
            ))
            audioRecord.startRecording()
        }
    }

    fun stopRecording() {
        audioRecord.stop()
        val id = communicationId!!
        // Sending remaining audio data
        var numOutBytes = writeAudioDataToSocket()
        while (numOutBytes > 0) {
            numOutBytes = writeAudioDataToSocket()
        }
        webSocketManager.connection.send(
            mapOf("status" to "FINISHED", "audio" to ByteArray(0), "id" to id)
        )
    }

    fun writeAudioDataToSocket(): Int {
        val id = communicationId!!
        val outBuffer = ByteArray(audioRecord.bufferSizeInFrames * 2)
        val numOutBytes = audioRecord.read(outBuffer, 0, outBuffer.size)
        if (numOutBytes > 0) {
            webSocketManager.connection.send(mapOf(
                "audio" to outBuffer.sliceArray(0..numOutBytes - 1),
                "id" to id,
                "status" to "RECORDING"
            ))
        }
        return numOutBytes
    }

    fun playAudio(audioData: ByteArray) {
        if (audioPlayer == null) {
            audioPlayer = AudioPlayer(externalScope)
        }
        audioPlayer!!.playAudio(audioData.toFloatArray())
    }

    fun stopPlayback() {
        if (audioPlayer != null) {
            audioPlayer!!.terminateAudioProcessing()
            audioPlayer = null
        }
    }

    fun setupAudioRecord(activity: MainActivity) {
         audioRecord = com.example.voicenoteclient.setupAudioRecord(
            activity, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
    }
}
