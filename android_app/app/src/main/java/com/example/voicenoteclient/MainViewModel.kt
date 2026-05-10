package com.example.voicenoteclient

import android.app.Application
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class UiState(
    val transcriptionText: String = "",
    val isRecording: Boolean = false,
    val isActionButtonsEnabled: Boolean = false,
    val savePath: String? = null,
    val serverSettings: ServerSettings = ServerSettings(),
    val connectionStatus: ConnectionStatus = ConnectionStatus.CONNECTING
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val repository = VoiceNoteRepository(application, viewModelScope)
    private val _uiState = MutableStateFlow(UiState(serverSettings = repository.serverSettings))
    val uiState = _uiState.asStateFlow()

    private var recordingJob: Job? = null

    init {
        repository.initialize()
        listenForMessages()
        listenForConnectionStatus()
    }

    private fun listenForConnectionStatus() {
        viewModelScope.launch {
            repository.connectionStatus.collect { status ->
                if (status != ConnectionStatus.CONNECTED && _uiState.value.isRecording) {
                    recordingJob?.cancel()
                }
                _uiState.update {
                    it.copy(
                        connectionStatus = status,
                        isRecording = if (status == ConnectionStatus.CONNECTED) it.isRecording else false
                    )
                }
            }
        }
    }

    private fun listenForMessages() {
        viewModelScope.launch {
            for (msg in repository.messageChannel) {
                when {
                    msg.containsKey(ProtocolKey.TEXT) -> {
                        val newText = _uiState.value.transcriptionText + msg[ProtocolKey.TEXT].toString()
                        val savePath = msg[ProtocolKey.SAVE_PATH] as? String ?: _uiState.value.savePath
                        _uiState.update { it.copy(transcriptionText = newText, savePath = savePath) }
                    }
                    msg[ProtocolKey.AUDIO] is ByteArray -> {
                        repository.playAudio(msg[ProtocolKey.AUDIO] as ByteArray)
                    }
                    else -> {
                        Log.d(AppLog.TAG, msg[ProtocolKey.STATUS].toString())
                    }
                }
            }
        }
    }

    fun onRecordButtonPress() {
        if (_uiState.value.isRecording || _uiState.value.connectionStatus != ConnectionStatus.CONNECTED) return

        _uiState.update { it.copy(isRecording = true, transcriptionText = "") }
        repository.stopPlayback()
        recordingJob?.cancel()
        recordingJob = viewModelScope.launch(Dispatchers.IO) {
            if (!repository.startRecording()) {
                _uiState.update { it.copy(isRecording = false) }
                return@launch
            }

            while (isActive && _uiState.value.isRecording) {
                repository.writeAudioDataToSocket()
            }
        }
    }

    fun onRecordButtonRelease() {
        if (!_uiState.value.isRecording) return

        _uiState.update { it.copy(isRecording = false, isActionButtonsEnabled = true) }
        recordingJob?.cancel()
        viewModelScope.launch(Dispatchers.IO) {
            repository.stopRecording()
        }
    }

    fun onDeleteButtonPress() {
        _uiState.update { it.copy(transcriptionText = "", isActionButtonsEnabled = false) }
        repository.sendAction(ConversationAction.DELETE, _uiState.value.savePath)
    }

    fun onNewChatButtonPress() {
        _uiState.update { it.copy(transcriptionText = "", isActionButtonsEnabled = false) }
        repository.sendAction(ConversationAction.NEW, _uiState.value.savePath)
    }

    fun onReconnectButtonPress() {
        repository.reconnect()
    }

    fun setupAudioRecord(activity: AppCompatActivity) {
        repository.setupAudioRecord(activity)
    }

    fun onSaveSettings(host: String, port: Int) {
        val settings = ServerSettings(host = host, port = port)
        repository.updateServerSettings(settings)
        _uiState.update { it.copy(serverSettings = repository.serverSettings) }
    }

    override fun onCleared() {
        repository.close()
        super.onCleared()
    }
}
