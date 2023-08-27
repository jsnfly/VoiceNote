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
import java.io.DataOutputStream
import java.io.DataInputStream
import java.net.Socket
import java.net.InetSocketAddress
import android.annotation.SuppressLint
import android.view.MotionEvent
import android.view.View
import Message
import sendMessage
import receiveMessage

class MainActivity : AppCompatActivity() {
    private val sampleRate = 44100
    private lateinit var audioRecord: AudioRecord
    private val audioConfig = mapOf("format" to 8, "channels" to 1, "rate" to sampleRate)

    private var socket = Socket()
    private lateinit var dataOutputStream: DataOutputStream
    private lateinit var dataInputStream: DataInputStream

    private var isStreaming: Boolean = false
    private lateinit var streamingThread: Thread
    private lateinit var response: Message

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        audioRecord = setupAudioRecord(
            this, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        setupButtons()
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupButtons() {
        val recordButton: Button = findViewById(R.id.recordButton)
        val deleteButton: Button = findViewById(R.id.deleteButton)
        val wrongButton: Button = findViewById(R.id.wrongButton)
        val settingsButton: Button = findViewById(R.id.settingsButton)

        recordButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                recordButton.alpha = 0.25F
                stream()
            } else if (event.action == MotionEvent.ACTION_UP) {
                isStreaming = false
                streamingThread.join()
                deleteButton.isEnabled = true
                wrongButton.isEnabled = true
                recordButton.alpha = 1.0F
            }
            false
        }

        deleteButton.setOnTouchListener {_, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                sendSingleMessage(
                    mapOf("action" to "delete", "save_path" to response["save_path"].toString())
                ) {
                    deleteButton.isEnabled = false
                    wrongButton.isEnabled = false
                    findViewById<TextView>(R.id.transcription).text = ""
                }
            }
            false
        }

        wrongButton.setOnTouchListener {_, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                sendSingleMessage(
                    mapOf("action" to "wrong", "save_path" to response["save_path"].toString())
                ) { wrongButton.isEnabled = false }
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

    private fun stream() {
        streamingThread = Thread {
            connect()
            sendMessage(
                mapOf("audio_config" to audioConfig, "topic" to findViewById<EditText>(R.id.editTextTopic).text.toString()),
                dataOutputStream
            )
            audioRecord.startRecording()
            isStreaming = true

            try {
                while (isStreaming) {
                    writeAudioDataToSocket()
                }
            } catch (e: Exception) {
                Log.e("ERROR", "Error while streaming audio.", e)
            }
            stopStreaming()
        }
        streamingThread.start()
    }

    private fun connect() {
        while (true) {
            Log.d("DEBUG", "Trying to connect...")
            try {
                socket.connect(InetSocketAddress(
                    findViewById<EditText>(R.id.editTextHost).text.toString(),
                    findViewById<EditText>(R.id.editTextPort).text.toString().toInt(),
                ))
                dataOutputStream = DataOutputStream(socket.getOutputStream())
                dataInputStream = DataInputStream(socket.getInputStream())
                break
            } catch (e: java.net.ConnectException) {
                Thread.sleep(200)

                // otherwise a `java.net.SocketException` is raised, that claims the socket is
                // closed (don't understand why).
                socket = Socket()
            }
        }
        Log.d("DEBUG", "Connected.")
    }

    private fun writeAudioDataToSocket(): Int {
        val outBuffer = ByteArray(audioRecord.bufferSizeInFrames * 2)
        val numOutBytes = audioRecord.read(outBuffer, 0, outBuffer.size)
        if (numOutBytes > 0) {
            dataOutputStream.write(outBuffer, 0, numOutBytes)
            // // Convert sample to hex:
            // val start = if (numOutBytes > 64) numOutBytes - 64 else 0
            // val hex = outBuffer.sliceArray(start until numOutBytes).joinToString(separator = " ") { String.format("%02X", it) }
            // Log.d("DEBUG", "After $noEmptyLoops sent $numOutBytes: $hex.")
        }
        return numOutBytes
    }

    private fun stopStreaming() {
        Log.d("DEBUG", "stopStreaming")
        audioRecord.stop()
        var numOutBytes = writeAudioDataToSocket()
        while (numOutBytes > 0) {
            numOutBytes = writeAudioDataToSocket()
        }
        dataOutputStream.flush()
        socket.shutdownOutput()

        response = receiveMessage(dataInputStream)
        findViewById<TextView>(R.id.transcription).text = response["text"].toString()

        dataOutputStream.close()
        dataInputStream.close()

        socket.close()
        socket = Socket()
    }

    private fun sendSingleMessage(msg: Map<String, String>, onSuccess: () -> Unit ) {
        val thread = Thread {
            connect()
            sendMessage(msg, dataOutputStream)
            socket.close()
            socket = Socket()
            runOnUiThread {
                onSuccess()
            }
        }
        thread.start()
        thread.join()
    }

    override fun onDestroy() {
        Log.d("DEBUG", "onDestroy")
        super.onDestroy()
    }

    override fun onPause() {
        Log.d("DEBUG", "onPause")
        super.onPause()
    }
}