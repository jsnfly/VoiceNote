package com.example.voicenoteclient

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.media.AudioFormat
import android.media.AudioRecord
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.EditText
import android.widget.LinearLayout
import android.annotation.SuppressLint
import android.view.MotionEvent
import android.view.View
import android.widget.CheckBox

import io.ktor.client.*
import io.ktor.client.features.websocket.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.http.cio.websocket.*
import kotlinx.coroutines.*
import kotlinx.coroutines.launch

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

fun ByteArray.toFloatArray(): FloatArray {
    val buffer = ByteBuffer.wrap(this)
    buffer.order(ByteOrder.LITTLE_ENDIAN) // Make sure to set the correct endianness
    val floatArray = FloatArray(this.size / 4)
    buffer.asFloatBuffer().get(floatArray)
    return floatArray
}

class MainActivity : AppCompatActivity() {
    private val sampleRate = 44100
    private val audioConfig = mapOf("format" to 8, "channels" to 1, "rate" to sampleRate)

    private lateinit var recordingThread: Thread
    private var isRecording: Boolean = false
    private lateinit var savePath: String

    private lateinit var audioRecord: AudioRecord
    private var audioPlayer: AudioPlayer? = null
    private lateinit var websocketManager: WebSocketManager

    private lateinit var recordButton: Button
    private lateinit var deleteButton: Button
    private lateinit var newChatButton: Button
    private lateinit var wrongButton: Button
    private lateinit var settingsButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        audioRecord = setupAudioRecord(
            this, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        websocketManager = WebSocketManager("192.168.0.154", 12345)

        setupButtons()
        GlobalScope.launch(Dispatchers.IO) {
            recvDataFromSocket()
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupButtons() {
        recordButton = findViewById(R.id.recordButton)
        deleteButton = findViewById(R.id.deleteButton)
        newChatButton = findViewById(R.id.newChatButton)
        wrongButton = findViewById(R.id.wrongButton)
        settingsButton = findViewById(R.id.settingsButton)

        recordButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                stopPlayback()
                recordButton.alpha = 0.25F
                findViewById<TextView>(R.id.transcription).text = ""
                startRecording()
            } else if (event.action == MotionEvent.ACTION_UP) {
                isRecording = false
                stopRecording()
                deleteButton.isEnabled = true
                wrongButton.isEnabled = true
                recordButton.alpha = 1.0F
            }
            false
        }

        deleteButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                sendAction("DELETE")
            }
            false
        }
        wrongButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                sendAction("WRONG")
            }
            false
        }
        newChatButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                sendAction("NEW CHAT")
            }
            false
        }

        val settingsLayout: LinearLayout = findViewById(R.id.settingsLayout)
        val saveButton: Button = findViewById(R.id.saveSettingsButton)
        settingsButton.setOnClickListener {
            if (settingsLayout.visibility == View.VISIBLE) {
                settingsLayout.visibility = View.GONE
            } else {
                settingsLayout.visibility = View.VISIBLE
            }
        }
        saveButton.setOnClickListener {
            settingsLayout.visibility = View.GONE
        }
    }

    private fun sendAction(action: String) {
        if (action != "WRONG") {
            deleteButton.isEnabled = false
            wrongButton.isEnabled = false
            findViewById<TextView>(R.id.transcription).text = ""
        }
        val id = UUID.randomUUID().toString()
        websocketManager.connection.reset(id)
        websocketManager.connection.send(mapOf(
            "action" to action,
            "id" to id,
            "save_path" to savePath,
            "status" to "INITIALIZING"
        ))
    }

    private fun startRecording() {
        recordingThread = Thread {
            val id = UUID.randomUUID().toString()
            websocketManager.connection.reset(id)
            websocketManager.connection.send(mapOf(
                "audio_config" to audioConfig,
                "chat_mode" to findViewById<CheckBox>(R.id.checkBoxChatMode).isChecked,
                "id" to id,
                "status" to "INITIALIZING",
                "topic" to findViewById<EditText>(R.id.editTextTopic).text.toString()
            ))
            audioRecord.startRecording()
            isRecording = true
            try {
                while (isRecording) { writeAudioDataToSocket(id) }
            } catch (e: Exception) {
                Log.e("ERROR", "Error while streaming audio.", e)
            }
        }
        recordingThread.start()
    }

    private fun writeAudioDataToSocket(id: String): Int {
        val outBuffer = ByteArray(audioRecord.bufferSizeInFrames * 2)
        val numOutBytes = audioRecord.read(outBuffer, 0, outBuffer.size)
        if (numOutBytes > 0) {
            websocketManager.connection.send(mapOf(
                "audio" to outBuffer.sliceArray(0..numOutBytes - 1),
                "id" to id,
                "status" to "RECORDING"
            ))
        }
        return numOutBytes
    }

    private suspend fun recvDataFromSocket() {
        while (true) {
            while (!websocketManager.isConnectionInitialized()) {
                delay(50)
            }
            val messages = websocketManager.connection.receive()
            for (msg in messages) {
                if (msg.containsKey("text")) {
                    savePath = msg["save_path"].toString()
                    findViewById<TextView>(R.id.transcription).text = findViewById<TextView>(R.id.transcription).text.toString() + msg["text"].toString()
                } else if (msg.containsKey("audio")) {
                    if (audioPlayer == null) { audioPlayer = AudioPlayer() }
                    audioPlayer!!.playAudio((msg["audio"] as ByteArray).toFloatArray())
                    // TODO: Finishing currently only works because tts gives empty array with its "FINISHED" message.
                } else {
                    Log.d("XXXXX", msg["status"].toString())
                }
            }
            delay(50)
        }
    }

    private fun stopRecording() {
        Log.d("XXXXX", "STOP RECORDING")
        audioRecord.stop()
        val id = websocketManager.connection.communicationID!!
        var numOutBytes = writeAudioDataToSocket(id)
        while (numOutBytes > 0) { numOutBytes = writeAudioDataToSocket(id) }
        websocketManager.connection.send(
            mapOf("status" to "FINISHED", "audio" to ByteArray(0), "id" to id)
        )
    }

    private fun stopPlayback() {
        if (audioPlayer != null) {
            audioPlayer!!.terminateAudioProcessing()
            audioPlayer = null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}