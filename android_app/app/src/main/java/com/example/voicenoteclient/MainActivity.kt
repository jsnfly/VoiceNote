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
import java.io.DataInputStream
import java.net.Socket
import java.net.InetSocketAddress
import Message

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val SAMPLE_RATE = 44100
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
    private var socket = Socket()
    private var audioRecord: AudioRecord? = null
    private var dataOutputStream: DataOutputStream? = null
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

        // Start streaming audio data to the server
        Thread {
            connect("192.168.0.154", 12345)

            try {
                while (isStreaming) {
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

    private fun connect(host: String, port: Int) {
        val address = InetSocketAddress(host, port)
        while (true) {
            Log.d("DEBUG", "Trying to connect...")
            try {
                socket.connect(address)
                dataOutputStream = DataOutputStream(socket!!.getOutputStream())
                // socket.soTimeout = 0
                // socket.tcpNoDelay = true
                // socket.keepAlive = true
                // socket.receiveBufferSize = 8192
                // socket.sendBufferSize = 8192
                // socket.reuseAddress = true
                // socket.setSoLinger(true, 0)
                // socket.setTcpNoDelay(true)
                break
            } catch (e: java.net.ConnectException) {
                Thread.sleep(1000)
                socket = Socket()  // otherwise a `java.net.SocketException` is raised, that claims the socket is closed
                                  // (don't understand why).
            }
        }
        Log.d("DEBUG", "Connected.")
        send_message(
            // TODO: Do not hardcode.
            mapOf(
                "format" to 8,
                "channels" to 1,
                "rate" to 44100
            )
        )
        val msg = resv_message()
        assert(msg["response"] == "OK")
    }

    fun send_message(data: Map<String, Any>) {
        dataOutputStream!!.write(Message(data).encode())
        dataOutputStream!!.flush()
    }

    fun resv_message(): Message {
        val buffer = ByteArray(1024)
        val bytesRead = DataInputStream(socket!!.getInputStream()).read(buffer)
        return Message.decode(buffer.sliceArray(0..bytesRead-1))
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