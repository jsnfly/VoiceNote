package com.example.voicenoteclient

import android.media.AudioFormat
import android.media.AudioTrack
import android.media.AudioManager

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
    private val sampleRate = 24000  // TODO: do not hardcode

    fun playAudio(audioData: FloatArray) {
        val shortArray = audioData.toShortArray()
        // Ensure the previous instance is stopped and released
        stopAudio()

        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        audioTrack = AudioTrack(
            AudioManager.STREAM_MUSIC,
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
            AudioTrack.MODE_STREAM
        )

        audioTrack?.play()
        isPlaying = true

        // Writing data to AudioTrack in chunks in a separate thread
        Thread {
            val chunkSize = 4096 // You can adjust this size
            var i = 0
            while (i < shortArray.size && isPlaying) {
                val end = minOf(i + chunkSize, shortArray.size)
                audioTrack?.write(shortArray, i, end - i)
                i += chunkSize
            }
            audioTrack?.stop()
            audioTrack?.release()
        }.start()
    }

    fun stopAudio() {
        isPlaying = false
        audioTrack?.apply {
            if (state == AudioTrack.STATE_INITIALIZED) {
                stop()
            }
            release()
        }
        audioTrack = null
    }
}
