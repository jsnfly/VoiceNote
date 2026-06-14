package com.example.voicenoteclient

object AppLog {
    const val TAG = "VoiceNoteClient"
}

object AppDefaults {
    const val HOST = "192.168.0.154"
    const val PORT = 12345
    const val INPUT_SAMPLE_RATE = 44100
    const val OUTPUT_SAMPLE_RATE = 48000
}

data class ServerSettings(
    val host: String = AppDefaults.HOST,
    val port: Int = AppDefaults.PORT
)

enum class ConnectionStatus {
    CONNECTING,
    CONNECTED,
    DISCONNECTED
}

object ProtocolKey {
    const val ACTION = "action"
    const val AUDIO = "audio"
    const val AUDIO_CONFIG = "audio_config"
    const val CHANNELS = "channels"
    const val FORMAT = "format"
    const val ID = "id"
    const val RATE = "rate"
    const val SAVE_PATH = "save_path"
    const val STATUS = "status"
    const val TEXT = "text"
}

object ProtocolStatus {
    const val INITIALIZING = "INITIALIZING"
    const val RECORDING = "RECORDING"
    const val FINISHED = "FINISHED"
    const val RESET = "RESET"
}

object ConversationAction {
    const val DELETE = "DELETE CONVERSATION"
    const val NEW = "NEW CONVERSATION"
}
