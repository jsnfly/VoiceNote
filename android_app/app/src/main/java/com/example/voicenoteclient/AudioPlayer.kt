package com.example.voicenoteclient

import android.media.AudioFormat
import android.media.AudioTrack
import android.media.AudioManager
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CoroutineScope

fun FloatArray.toShortArray(): ShortArray {
    return this.map { sample ->
        // Clamp the value to the range [-1.0, 1.0]
        val clampedSample = sample.coerceIn(-1.0f, 1.0f)

        // Scale and convert to a short (16-bit integer)
        (clampedSample * Short.MAX_VALUE).toInt().toShort()
    }.toShortArray()
}
class AudioPlayer(private val scope: CoroutineScope) {

    private var audioTrack: AudioTrack? = null
    private var isPlaying = false
    private val sampleRate = 24000  // TODO: Consider moving this to a configuration or dynamic setting
    private val audioQueue: Channel<ShortArray> = Channel(Channel.UNLIMITED)
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
        startAudioJob()
    }

    private fun startAudioJob() {
        scope.launch(Dispatchers.IO) {
            for (data in audioQueue) {
                if (isPlaying) {
                    if (data.isEmpty()) break
                    audioTrack?.write(data, 0, data.size)
                }
            }
        }
    }

    fun playAudio(audioData: FloatArray) {
        val shortArray = audioData.toShortArray()
        scope.launch {
            audioQueue.send(shortArray)
        }
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
        scope.launch {
            audioQueue.send(shortArrayOf()) // Signal to terminate
        }
        audioQueue.close()
    }
}