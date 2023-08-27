package com.example.voicenoteclient

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat

@Suppress("SameParameterValue")
fun setupAudioRecord(
    activity: AppCompatActivity, sampleRate: Int, channelConfig: Int, audioEncoding: Int
): AudioRecord {
    if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(activity, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
    }

    val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioEncoding)
    return AudioRecord(
        MediaRecorder.AudioSource.MIC,
        sampleRate,
        channelConfig,
        audioEncoding,
        50 * minBufferSize
    )
}