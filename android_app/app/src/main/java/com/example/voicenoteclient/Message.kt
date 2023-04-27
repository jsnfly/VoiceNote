import com.google.gson.Gson

class Message(private val data: Map<String, Any>) {

    companion object {
        private const val SEP = "\n\n\n\n\n\n\n\n\n"

        fun decode(bytes: ByteArray): Message {
            val json = bytes.toString(Charsets.UTF_8).replace(SEP, "")
            val data = Gson().fromJson(json, Map::class.java)
            return Message(data as Map<String, Any>)
        }

        private fun stringify(value: Any?): Any? {
            return when (value) {
                is String -> value
                is ByteArray -> String(value)
                is Map<*, *> -> value.mapValues { stringify(it.value) }
                else -> value
            }
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
        val json = Gson().toJson(stringify(data))
        return "$SEP$json$SEP".toByteArray(Charsets.UTF_8)
    }

    operator fun contains(key: String): Boolean {
        return data.containsKey(key)
    }

    operator fun get(key: String): Any? {
        return data[key]
    }
}
