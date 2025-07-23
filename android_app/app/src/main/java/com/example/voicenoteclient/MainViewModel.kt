package com.example.voicenoteclient

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class UiState(
    val transcriptionText: String = "",
    val isRecording: Boolean = false,
    val isActionButtonsEnabled: Boolean = false,
    val savePath: String? = null,
    val isChatMode: Boolean = true,
    val topic: String = "misc"
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(UiState())
    val uiState = _uiState.asStateFlow()

    private val repository = VoiceNoteRepository(application, viewModelScope)
    private var recordingJob: Job? = null

    init {
        repository.initialize()
        listenForMessages()
    }

    private fun listenForMessages() {
        viewModelScope.launch {
            for (msg in repository.messageChannel) {
                if (msg.containsKey("text")) {
                    val newText = _uiState.value.transcriptionText + msg["text"].toString()
                    _uiState.update { it.copy(transcriptionText = newText, savePath = msg["save_path"].toString()) }
                } else if (msg.containsKey("audio")) {
                    repository.playAudio(msg["audio"] as ByteArray)
                } else {
                    Log.d("XXXXX", msg["status"].toString())
                }
            }
        }
    }

    fun onRecordButtonPress() {
        _uiState.update { it.copy(isRecording = true, transcriptionText = "") }
        repository.stopPlayback()
        repository.startRecording(_uiState.value.isChatMode, _uiState.value.topic)
        recordingJob = viewModelScope.launch {
            while (_uiState.value.isRecording) {
                repository.writeAudioDataToSocket()
            }
        }
    }

    fun onRecordButtonRelease() {
        _uiState.update { it.copy(isRecording = false, isActionButtonsEnabled = true) }
        recordingJob?.cancel()
        repository.stopRecording()
    }

    fun onDeleteButtonPress() {
        _uiState.update { it.copy(transcriptionText = "", isActionButtonsEnabled = false) }
        repository.sendAction("DELETE", _uiState.value.savePath)
    }

    fun onWrongButtonPress() {
        repository.sendAction("WRONG", _uiState.value.savePath)
    }

    fun onNewChatButtonPress() {
        _uiState.update { it.copy(transcriptionText = "", isActionButtonsEnabled = false) }
        repository.sendAction("NEW CHAT", _uiState.value.savePath)
    }

    fun onChatModeChange(isChatMode: Boolean) {
        _uiState.update { it.copy(isChatMode = isChatMode) }
    }

    fun onTopicChange(topic: String) {
        _uiState.update { it.copy(topic = topic) }
    }

    fun setupAudioRecord(activity: MainActivity) {
        repository.setupAudioRecord(activity)
    }
}
