package com.example.voicenoteclient

import android.content.Context
import android.widget.Toast
import java.io.File
import java.io.FileWriter
import java.io.IOException

class CrashLogger(private val context: Context) : Thread.UncaughtExceptionHandler {
    private val defaultExceptionHandler = Thread.getDefaultUncaughtExceptionHandler()

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        try {
            val logFile = File(context.getExternalFilesDir(null), "crash_log.txt")
            FileWriter(logFile, true).use { writer ->
                writer.append("${System.currentTimeMillis()}: CRASH in thread ${thread.name}\n")
                writer.append("Error: ${throwable.message}\n")
                writer.append("Stack trace:\n")
                throwable.stackTrace.forEach { element ->
                    writer.append("    $element\n")
                }
                writer.append("\n")
            }
        } catch (_: IOException) {
            Toast.makeText(context, "Failed to save log", Toast.LENGTH_LONG).show()
        } finally {
            defaultExceptionHandler?.uncaughtException(thread, throwable)
        }
    }
}
