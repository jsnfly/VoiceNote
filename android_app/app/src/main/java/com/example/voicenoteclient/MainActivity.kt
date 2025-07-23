package com.example.voicenoteclient

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.addTextChangedListener
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder


fun ByteArray.toFloatArray(): FloatArray {
    val buffer = ByteBuffer.wrap(this)
    buffer.order(ByteOrder.LITTLE_ENDIAN) // Make sure to set the correct endianness
    val floatArray = FloatArray(this.size / 4)
    buffer.asFloatBuffer().get(floatArray)
    return floatArray
}

class CustomExceptionHandler(private val context: Context) : Thread.UncaughtExceptionHandler {
    private val defaultExceptionHandler = Thread.getDefaultUncaughtExceptionHandler()

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        try {
            // Log the crash
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
        } catch (e: IOException) {
            Toast.makeText(context, "Failed to save log", Toast.LENGTH_LONG).show()
        } finally {
            // Make sure to call the default handler after logging
            defaultExceptionHandler?.uncaughtException(thread, throwable)
        }
    }
}

class MainActivity : AppCompatActivity() {
    private val AUDIO_PERMISSION_REQUEST_CODE = 123

    private lateinit var recordButton: Button
    private lateinit var deleteButton: Button
    private lateinit var newChatButton: Button
    private lateinit var wrongButton: Button
    private lateinit var settingsButton: Button
    private lateinit var transcriptionView: TextView
    private lateinit var chatModeCheckBox: CheckBox
    private lateinit var topicEditText: EditText

    private val viewModel: MainViewModel by viewModels {
        MainViewModelFactory(application)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        Thread.setDefaultUncaughtExceptionHandler(CustomExceptionHandler(applicationContext))

        setupUI()
        observeViewModel()

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            viewModel.setupAudioRecord(this)
        } else {
            requestPermissions(
                arrayOf(Manifest.permission.RECORD_AUDIO),
                AUDIO_PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun observeViewModel() {
        lifecycleScope.launch {
            viewModel.uiState.collect { state ->
                transcriptionView.text = state.transcriptionText
                deleteButton.isEnabled = state.isActionButtonsEnabled
                wrongButton.isEnabled = state.isActionButtonsEnabled
                recordButton.alpha = if (state.isRecording) 0.25f else 1.0f
                if (chatModeCheckBox.isChecked != state.isChatMode) {
                    chatModeCheckBox.isChecked = state.isChatMode
                }
                if (topicEditText.text.toString() != state.topic) {
                    topicEditText.setText(state.topic)
                }
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            AUDIO_PERMISSION_REQUEST_CODE -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    viewModel.setupAudioRecord(this)
                } else {
                    Toast.makeText(
                        this,
                        "Audio recording permission is required for this feature",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupUI() {
        recordButton = findViewById(R.id.recordButton)
        deleteButton = findViewById(R.id.deleteButton)
        newChatButton = findViewById(R.id.newChatButton)
        wrongButton = findViewById(R.id.wrongButton)
        settingsButton = findViewById(R.id.settingsButton)
        transcriptionView = findViewById(R.id.transcription)
        chatModeCheckBox = findViewById(R.id.checkBoxChatMode)
        topicEditText = findViewById(R.id.editTextTopic)

        transcriptionView.movementMethod = ScrollingMovementMethod()

        recordButton.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                viewModel.onRecordButtonPress()
            } else if (event.action == MotionEvent.ACTION_UP) {
                viewModel.onRecordButtonRelease()
            }
            false
        }

        deleteButton.setOnClickListener { viewModel.onDeleteButtonPress() }
        wrongButton.setOnClickListener { viewModel.onWrongButtonPress() }
        newChatButton.setOnClickListener { viewModel.onNewChatButtonPress() }

        chatModeCheckBox.setOnCheckedChangeListener { _, isChecked ->
            viewModel.onChatModeChange(isChecked)
        }

        topicEditText.addTextChangedListener { text ->
            viewModel.onTopicChange(text.toString())
        }

        val settingsLayout: LinearLayout = findViewById(R.id.settingsLayout)
        val saveButton: Button = findViewById(R.id.saveSettingsButton)
        settingsButton.setOnClickListener {
            if (settingsLayout.visibility == View.VISIBLE) {
                settingsLayout.visibility = View.GONE
            } else {
                settingsLayout.visibility = View.VISIBLE
            }
        }
        saveButton.setOnClickListener {
            settingsLayout.visibility = View.GONE
        }
    }
}