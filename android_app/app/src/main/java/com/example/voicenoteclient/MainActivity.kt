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
import android.text.method.ScrollingMovementMethod

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

import android.content.Context
import java.io.File
import java.io.FileWriter
import java.io.IOException
import android.widget.Toast

import android.content.pm.PackageManager

fun ByteArray.toFloatArray(): FloatArray {
    val buffer = ByteBuffer.wrap(this)
    buffer.order(ByteOrder.LITTLE_ENDIAN) // Make sure to set the correct endianness
    val floatArray = FloatArray(this.size / 4)
    buffer.asFloatBuffer().get(floatArray)
    return floatArray
}

class CustomExceptionHandler(private val context: Context) : Thread.UncaughtExceptionHandler {
    private val defaultExceptionHandler = Thread.getDefaultUncaughtExceptionHandler()

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        try {
            // Log the crash
            val logFile = File(context.getExternalFilesDir(null), "crash_log.txt")
            FileWriter(logFile, true).use { writer ->
                writer.append("${System.currentTimeMillis()}: CRASH in thread ${thread.name}\n")
                writer.append("Error: ${throwable.message}\n")
                writer.append("Stack trace:\n")
                throwable.stackTrace.forEach { element ->
                    writer.append("    $element\n")
                }
                writer.append("\n")
            }
        } catch (e: IOException) {
            Toast.makeText(context, "Failed to save log", Toast.LENGTH_LONG).show()
        } finally {
            // Make sure to call the default handler after logging
            defaultExceptionHandler?.uncaughtException(thread, throwable)
        }
    }
}

class MainActivity : AppCompatActivity() {
    private val AUDIO_PERMISSION_REQUEST_CODE = 123

    private val sampleRate = 44100
    private val audioConfig = mapOf("format" to 8, "channels" to 1, "rate" to sampleRate)

    private lateinit var recordingThread: Thread
    private var isRecording: Boolean = false
    private var savePath: String? = null

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
        Thread.setDefaultUncaughtExceptionHandler(CustomExceptionHandler(applicationContext))

        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            audioRecord = setupAudioRecord(
                this, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
        } else {
            // Request the permission
            requestPermissions(
                arrayOf(android.Manifest.permission.RECORD_AUDIO),
                AUDIO_PERMISSION_REQUEST_CODE
            )
        }

        websocketManager = WebSocketManager("192.168.0.154", 12345)

        setupButtons()
        findViewById<TextView>(R.id.transcription).movementMethod = ScrollingMovementMethod()
        GlobalScope.launch(Dispatchers.IO) {
            recvDataFromSocket()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            AUDIO_PERMISSION_REQUEST_CODE -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // Permission was granted
                    audioRecord = setupAudioRecord(
                        this, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
                    )
                } else {
                    // Permission denied
                    Toast.makeText(
                        this,
                        "Audio recording permission is required for this feature",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
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