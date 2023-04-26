package com.example.voicenoteclient

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import android.widget.Button
import androidx.core.app.ActivityCompat
import java.io.DataOutputStream
import java.net.Socket

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val SAMPLE_RATE = 44100
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
    private var audioRecord: AudioRecord? = null
    private var dataOutputStream: DataOutputStream? = null
    private var socket: Socket? = null
    private var isStreaming: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Set up audio recording
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            BUFFER_SIZE
        )

        // Set up stream button
        val streamButton: Button = findViewById(R.id.streamButton)
        streamButton.setOnClickListener {
            if (!isStreaming) {
                startStreaming()
                streamButton.setText(R.string.stop_streaming)
            } else {
                stopStreaming()
                streamButton.setText(R.string.start_streaming)
            }
        }
    }

    private fun startStreaming() {
        isStreaming = true
        audioRecord!!.startRecording()

        Log.d("myTag", "This is my message")

        // Start streaming audio data to the server
        Thread {
            try {
                while (isStreaming) {
                    // Set up socket connection
                    val address = "192.168.0.154"
                    val port = 12345 // Replace with your server port
                    socket = Socket(address, port)
                    dataOutputStream = DataOutputStream(socket!!.getOutputStream())

                    val buffer = ByteArray(BUFFER_SIZE)
                    val bytesRead = audioRecord!!.read(buffer, 0, buffer.size)
                    if (bytesRead > 0) {
                        dataOutputStream!!.write(buffer, 0, bytesRead)
                        dataOutputStream!!.flush()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error while streaming audio.", e)
            }
        }.start()
    }

    private fun stopStreaming() {
        isStreaming = false
        audioRecord!!.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        audioRecord?.release()
        socket?.close()
        dataOutputStream?.close()
    }
}