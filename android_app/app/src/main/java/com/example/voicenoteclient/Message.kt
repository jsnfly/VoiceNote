import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.util.Base64

data class Message(val data: Map<String, Any?>) {
    companion object {
        private val gson = Gson()

        fun fromDataString(dataString: String): Message {
            val type = object : TypeToken<Map<String, Any?>>() {}.type
            val data: Map<String, Any?> = gson.fromJson(dataString, type)
            return Message(destringifyValues(data))
        }

        private fun destringifyValues(encodedData: Map<String, Any?>): Map<String, Any?> {
            val transformed = mutableMapOf<String, Any?>()
            encodedData.forEach { (key, value) ->
                transformed[key.removeSuffix("_base64")] = when {
                    key.endsWith("_base64") && value is String -> Base64.getDecoder().decode(value)
                    value is Map<*, *> -> destringifyValues(value as Map<String, Any?>)
                    else -> value
                }
            }
            return transformed
        }
    }

    fun encode(): String {
        val stringifiedData = stringifyValues(data)
        return gson.toJson(stringifiedData)
    }

    private fun stringifyValues(data: Map<String, Any?>): Map<String, Any?> {
        val transformed = mutableMapOf<String, Any?>()
        data.forEach { (key, value) ->
            when (value) {
                is ByteArray -> transformed["${key}_base64"] = Base64.getEncoder().encodeToString(value)
                is Map<*, *> -> transformed[key] = stringifyValues(value as Map<String, Any?>)
                else -> transformed[key] = value
            }
        }
        return transformed
    }
}
