package com.example.voicenoteclient

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

class AudioPlayer(private val scope: CoroutineScope) {

    private var audioTrack: AudioTrack? = null
    private var isPlaying = false
    private val sampleRate = AppDefaults.OUTPUT_SAMPLE_RATE
    private val audioQueue: Channel<ShortArray> = Channel(Channel.UNLIMITED)
    private val bufferSize = AudioTrack.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_OUT_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    init {
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
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
