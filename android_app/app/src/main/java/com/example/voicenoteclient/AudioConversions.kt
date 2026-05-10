package com.example.voicenoteclient

import java.nio.ByteBuffer
import java.nio.ByteOrder

fun ByteArray.toFloatArray(): FloatArray {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val floatArray = FloatArray(size / 4)
    buffer.asFloatBuffer().get(floatArray)
    return floatArray
}

fun FloatArray.toShortArray(): ShortArray {
    return map { sample ->
        val clampedSample = sample.coerceIn(-1.0f, 1.0f)
        (clampedSample * Short.MAX_VALUE).toInt().toShort()
    }.toShortArray()
}
