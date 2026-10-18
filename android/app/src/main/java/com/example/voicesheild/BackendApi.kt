package com.example.voicesheild

import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/** Demo API only. No Twilio credentials belong in the Android app. */
object BackendApi {
    fun baseUrl(value: String): String {
        val url = URL(value.trim().trimEnd('/'))
        require(url.protocol == "https" && url.host.isNotBlank() && url.userInfo == null &&
            url.query == null && url.ref == null && (url.path.isEmpty() || url.path == "/")) {
            "Enter the HTTPS server address without /demo or another path."
        }
        return url.toString().trimEnd('/')
    }

    suspend fun request(base: String, path: String, body: JSONObject? = null): JSONObject =
        withContext(Dispatchers.IO) {
            val connection = URL(baseUrl(base) + path).openConnection() as HttpURLConnection
            try {
                connection.connectTimeout = 10000
                connection.readTimeout = 15000
                connection.instanceFollowRedirects = false
                connection.setRequestProperty("Accept", "application/json")
                if (body != null) {
                    connection.requestMethod = "POST"
                    connection.doOutput = true
                    connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                    connection.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
                }
                val status = connection.responseCode
                check(status in 200..299) { "Server returned HTTP $status. Check the server URL and demo configuration." }
                JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
            } finally {
                connection.disconnect()
            }
        }
}
