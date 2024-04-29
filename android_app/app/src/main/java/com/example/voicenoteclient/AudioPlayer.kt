package com.example.voicenoteclient

import android.media.AudioFormat
import android.media.AudioTrack
import android.media.AudioManager
import java.util.concurrent.BlockingQueue
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread
import android.util.Log

fun FloatArray.toShortArray(): ShortArray {
    return this.map { sample ->
        // Clamp the value to the range [-1.0, 1.0]
        val clampedSample = sample.coerceIn(-1.0f, 1.0f)

        // Scale and convert to a short (16-bit integer)
        (clampedSample * Short.MAX_VALUE).toInt().toShort()
    }.toShortArray()
}
class AudioPlayer {

    private var audioTrack: AudioTrack? = null
    private var isPlaying = false
    private val sampleRate = 24000  // TODO: Consider moving this to a configuration or dynamic setting
    private val audioQueue: BlockingQueue<ShortArray> = LinkedBlockingQueue()
    private val bufferSize = AudioTrack.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_OUT_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    init {
        audioTrack = AudioTrack(
            AudioManager.STREAM_MUSIC,
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
            AudioTrack.MODE_STREAM
        )
        startAudioThread()
    }

    private fun startAudioThread() {
        thread(start = true) {
            while (true) {
                try {
                    if (isPlaying) {
                        val data = audioQueue.take()
                        val length = data.size
                        Log.d("XXXXX", "Length2 $length")
                        if (data.isEmpty()) break  // Empty array can signal to terminate the thread

                        audioTrack?.write(data, 0, data.size)
                    }
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                }
            }
        }
    }

    fun playAudio(audioData: FloatArray) {
        val shortArray = audioData.toShortArray()
        val length = shortArray.size
        Log.d("XXXXX", "Length1 $length")
        audioQueue.put(shortArray)
        if (!isPlaying) {
            audioTrack?.play()
            isPlaying = true
        }
    }

    fun stopAudio() {
        isPlaying = false
        audioTrack?.apply {
            if (playState == AudioTrack.PLAYSTATE_PLAYING) {
                stop()
            }
        }
        releaseAudio()
    }

    fun releaseAudio() {
        audioTrack?.release()
        audioTrack = null
    }

    fun terminateAudioProcessing() {
        stopAudio()
        audioQueue.put(shortArrayOf())  // Use an empty array to signal the thread to exit
    }
}