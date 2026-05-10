package com.example.voicenoteclient

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var recordButton: Button
    private lateinit var deleteButton: Button
    private lateinit var newChatButton: Button
    private lateinit var settingsButton: Button
    private lateinit var reconnectButton: Button
    private lateinit var hostInput: EditText
    private lateinit var portInput: EditText
    private lateinit var connectionStatusView: TextView
    private lateinit var recordingHintView: TextView
    private lateinit var transcriptionView: TextView

    private val viewModel: MainViewModel by viewModels {
        MainViewModelFactory(application)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        Thread.setDefaultUncaughtExceptionHandler(CrashLogger(applicationContext))

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
                val isConnected = state.connectionStatus == ConnectionStatus.CONNECTED
                connectionStatusView.text = connectionStatusText(state.connectionStatus)
                connectionStatusView.setTextColor(connectionStatusColor(state.connectionStatus))
                recordingHintView.text = when {
                    state.isRecording -> getString(R.string.recording)
                    isConnected -> getString(R.string.hold_to_record)
                    else -> getString(R.string.waiting_for_server)
                }
                deleteButton.isEnabled = state.isActionButtonsEnabled && isConnected
                newChatButton.isEnabled = isConnected
                recordButton.isEnabled = isConnected
                reconnectButton.isEnabled = state.connectionStatus != ConnectionStatus.CONNECTING
                recordButton.alpha = when {
                    state.isRecording -> 0.35f
                    isConnected -> 1.0f
                    else -> 0.55f
                }

                if (!hostInput.hasFocus() && hostInput.text.toString() != state.serverSettings.host) {
                    hostInput.setText(state.serverSettings.host)
                }
                if (!portInput.hasFocus() && portInput.text.toString() != state.serverSettings.port.toString()) {
                    portInput.setText(state.serverSettings.port.toString())
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
        newChatButton = findViewById(R.id.newConversationButton)
        settingsButton = findViewById(R.id.settingsButton)
        reconnectButton = findViewById(R.id.reconnectButton)
        transcriptionView = findViewById(R.id.transcription)
        connectionStatusView = findViewById(R.id.connectionStatus)
        recordingHintView = findViewById(R.id.recordingHint)
        hostInput = findViewById(R.id.editTextHost)
        portInput = findViewById(R.id.editTextPort)

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
        newChatButton.setOnClickListener { viewModel.onNewChatButtonPress() }
        reconnectButton.setOnClickListener { viewModel.onReconnectButtonPress() }

        val settingsLayout: LinearLayout = findViewById(R.id.settingsLayout)
        val saveButton: Button = findViewById(R.id.saveSettingsButton)
        settingsButton.setOnClickListener {
            settingsLayout.visibility = if (settingsLayout.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        saveButton.setOnClickListener {
            val host = hostInput.text.toString().trim()
            val port = portInput.text.toString().toIntOrNull()
            if (host.isBlank() || port == null || port !in 1..65535) {
                Toast.makeText(this, "Enter a valid host and port", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }

            viewModel.onSaveSettings(host, port)
            settingsLayout.visibility = View.GONE
        }
    }

    private companion object {
        const val AUDIO_PERMISSION_REQUEST_CODE = 123
    }

    private fun connectionStatusText(status: ConnectionStatus): String {
        return when (status) {
            ConnectionStatus.CONNECTING -> getString(R.string.connection_connecting)
            ConnectionStatus.CONNECTED -> getString(R.string.connection_connected)
            ConnectionStatus.DISCONNECTED -> getString(R.string.connection_disconnected)
        }
    }

    private fun connectionStatusColor(status: ConnectionStatus): Int {
        return when (status) {
            ConnectionStatus.CONNECTING -> Color.rgb(141, 103, 18)
            ConnectionStatus.CONNECTED -> Color.rgb(35, 112, 68)
            ConnectionStatus.DISCONNECTED -> Color.rgb(176, 42, 55)
        }
    }
}
