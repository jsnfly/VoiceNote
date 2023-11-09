import com.google.gson.Gson
import java.io.DataOutputStream
import java.io.DataInputStream
import java.io.ByteArrayOutputStream
import java.net.SocketException

class Message(private val data: Map<String, Any>) {

    companion object {
        private const val SEP = "\n\n\n\n\n\n\n\n\n"

        fun decode(bytes: ByteArray): Message {
            val json = bytes.toString(Charsets.UTF_8).replace(SEP, "")
            val data = destringify(Gson().fromJson(json, Map::class.java))
            return Message(data as Map<String, Any>)
        }

        private fun destringify(value: Any?): Any? {
            return when (value) {
                is String -> {
                    try {
                        value.toInt()
                    } catch (e: NumberFormatException) {
                        try {
                            value.toDouble()
                        } catch (e: NumberFormatException) {
                            value
                        }
                    }
                }
                is Map<*, *> -> value.mapValues { destringify(it.value) }
                else -> value
            }
        }
    }

    fun encode(): ByteArray {
        val json = Gson().toJson(data)
        return "$SEP$json$SEP".toByteArray(Charsets.UTF_8)
    }

    operator fun contains(key: String): Boolean {
        return data.containsKey(key)
    }

    operator fun get(key: String): Any? {
        return data[key]
    }
}

fun sendMessage(data: Map<String, Any>, dataOutputStream: DataOutputStream) {
    dataOutputStream.write(Message(data).encode())
    dataOutputStream.flush()
}

fun receiveMessage(dataInputStream: DataInputStream): Message {
    val bufferSize = 1024 * 1024
    val buffer = ByteArray(bufferSize)
    val byteOutputStream = ByteArrayOutputStream()

    var bytesRead: Int
    while (true) {
        try {
            bytesRead = dataInputStream.read(buffer)
        } catch (e: SocketException) {
            break
        }
        if (bytesRead == -1) {
            // End of stream reached
            break
        }
        byteOutputStream.write(buffer, 0, bytesRead)
    }
    val totalData = byteOutputStream.toByteArray()
    return Message.decode(totalData)
}
