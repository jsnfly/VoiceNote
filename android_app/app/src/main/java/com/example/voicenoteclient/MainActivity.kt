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
import android.widget.TextView
import androidx.core.app.ActivityCompat
import java.io.DataOutputStream
import java.io.DataInputStream
import java.net.Socket
import java.net.InetSocketAddress
import Message
import android.annotation.SuppressLint
import android.view.MotionEvent

class MainActivity : AppCompatActivity() {
    private val sampleRate = 44100
    private var audioRecord: AudioRecord? = null
    private val audioConfig = mapOf("format" to 8, "channels" to 1, "rate" to sampleRate)

    private val address = InetSocketAddress("192.168.0.154", 12345)
    private var socket = Socket()
    private var dataOutputStream: DataOutputStream? = null
    private var dataInputStream: DataInputStream? = null

    private var streamingThread: Thread? = null
    private var isStreaming: Boolean = false
    private var textOutput: TextView? = null
    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val recordButton: Button = findViewById(R.id.recordButton)
        textOutput = findViewById(R.id.transcription)

        setUpAudioRecord(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)

        recordButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                recordButton.alpha = 0.25F
                stream()
            } else if (event.action == MotionEvent.ACTION_UP) {
                isStreaming = false
                streamingThread!!.join()
                streamingThread = null
            }
            false
        }
    }

    private fun stream() {
        streamingThread = Thread {
            connect()
            audioRecord!!.startRecording()
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
        streamingThread!!.start()
    }

    private fun connect() {
        while (true) {
            Log.d("DEBUG", "Trying to connect...")
            try {
                socket.connect(address)
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
        sendMessage(audioConfig)
        assert(recvMessage()["response"] == "OK")
    }

    private fun sendMessage(data: Map<String, Any>) {
        dataOutputStream!!.write(Message(data).encode())
        dataOutputStream!!.flush()
    }

    private fun recvMessage(): Message {
        val buffer = ByteArray(4096)
        val bytesRead = dataInputStream!!.read(buffer)
        return Message.decode(buffer.sliceArray(0 until bytesRead))
    }

    @Suppress("SameParameterValue")
    private fun setUpAudioRecord(sampleRate: Int, channelConfig: Int, audioEncoding: Int) {
        if (ActivityCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }

        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioEncoding)
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioEncoding,
            50 * minBufferSize
        )
    }

    private fun writeAudioDataToSocket(): Int {
        val outBuffer = ByteArray(audioRecord!!.bufferSizeInFrames * 2)
        val numOutBytes = audioRecord!!.read(outBuffer, 0, outBuffer.size)
        if (numOutBytes > 0) {
            dataOutputStream!!.write(outBuffer, 0, numOutBytes)
            // // Convert sample to hex:
            // val start = if (numOutBytes > 64) numOutBytes - 64 else 0
            // val hex = outBuffer.sliceArray(start until numOutBytes).joinToString(separator = " ") { String.format("%02X", it) }
            // Log.d("DEBUG", "After $noEmptyLoops sent $numOutBytes: $hex.")
        }
        return numOutBytes
    }

    private fun stopStreaming() {
        Log.d("DEBUG", "stopStreaming")
        audioRecord!!.stop()
        var numOutBytes = writeAudioDataToSocket()
        while (numOutBytes > 0) {
            numOutBytes = writeAudioDataToSocket()
        }

        audioRecord!!.release()
        dataOutputStream!!.flush()

        socket.shutdownOutput()

        val text = recvMessage()["text"]
        textOutput!!.text = text.toString()

        dataOutputStream!!.close()
        dataInputStream!!.close()

        socket.close()
        socket = Socket()
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