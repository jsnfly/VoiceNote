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
    private val SAMPLE_RATE = 44100
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val MIN_BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
    private var socket = Socket()
    private var audioRecord: AudioRecord? = null
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

        recordButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                recordButton.setBackgroundColor(getResources().getColor(R.color.black));
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
            connect("192.168.0.154", 12345)
            setUpAudioRecord()
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

    private fun connect(host: String, port: Int) {
        val address = InetSocketAddress(host, port)
        while (true) {
            Log.d("DEBUG", "Trying to connect...")
            try {
                socket.connect(address)
                dataOutputStream = DataOutputStream(socket.getOutputStream())
                dataInputStream = DataInputStream(socket.getInputStream())
                break
            } catch (e: java.net.ConnectException) {
                Thread.sleep(200)

                // otherwise a `java.net.SocketException` is raised, that claims
                // the socket is closed (don't understand why).
                socket = Socket()
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
        val msg = recv_message()
        assert(msg["response"] == "OK")
    }

    fun send_message(data: Map<String, Any>) {
        dataOutputStream!!.write(Message(data).encode())
        dataOutputStream!!.flush()
    }

    fun recv_message(): Message {
        val buffer = ByteArray(4096)
        val bytesRead = dataInputStream!!.read(buffer)
        return Message.decode(buffer.sliceArray(0..bytesRead-1))
    }

    private fun setUpAudioRecord() {
        if (ActivityCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            50 * MIN_BUFFER_SIZE
        )
    }

    private fun writeAudioDataToSocket(): Int {
        val outBuffer = ByteArray(10 * MIN_BUFFER_SIZE)
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
        Log.d("DEBUGX", "stopStreaming")
        audioRecord!!.stop()
        var numOutBytes = writeAudioDataToSocket()
        while (numOutBytes > 0) {
            numOutBytes = writeAudioDataToSocket()
        }

        audioRecord!!.release()
        audioRecord = null

        dataOutputStream!!.flush()

        socket.shutdownOutput()

        val text = recv_message()["text"]
        Log.d("DEBUGX", "$text")
        textOutput!!.setText(text.toString())

        dataOutputStream!!.close()
        dataInputStream!!.close()

        socket.close()
        socket = Socket()
    }

    override fun onDestroy() {
        Log.d("DEBUGX", "onDestroy")
        super.onDestroy()
    }

    override fun onPause() {
        Log.d("DEBUGX", "onPause")
        super.onPause()
    }
}