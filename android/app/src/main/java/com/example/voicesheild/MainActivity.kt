package com.example.voicesheild

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import kotlinx.coroutines.CancellationException
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.material3.OutlinedTextField
import android.content.Intent
import android.net.Uri
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val Background = Color(0xFFF8FAFC)
private val White = Color(0xFFFFFFFF)
private val Primary = Color(0xFF2563EB)
private val DarkText = Color(0xFF111827)
private val GrayText = Color(0xFF64748B)
private val LightGray = Color(0xFFE2E8F0)

private val Green = Color(0xFF16A34A)
private val LightGreen = Color(0xFFDCFCE7)

private val Red = Color(0xFFDC2626)
private val LightRed = Color(0xFFFEE2E2)

data class Detection(
    val result: String,
    val confidence: Int,
    val time: String,
    val message: String
)

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Background
                ) {
                    VoiceShieldApp()
                }
            }
        }
    }
}

private fun vibratePhone(context: Context) {
    try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager =
                context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE)
                        as VibratorManager

            vibratorManager.defaultVibrator.vibrate(
                VibrationEffect.createOneShot(
                    300,
                    VibrationEffect.DEFAULT_AMPLITUDE
                )
            )
        } else {
            @Suppress("DEPRECATION")
            val vibrator =
                context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

            @Suppress("DEPRECATION")
            vibrator.vibrate(300)
        }
    } catch (_: Exception) {
        // Prevent vibration problems from crashing the app.
    }
}

@Composable
fun VoiceShieldApp() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val preferences = remember { context.getSharedPreferences("truetone", Context.MODE_PRIVATE) }
    var server by remember { mutableStateOf(preferences.getString("server", "") ?: "") }
    var currentPage by remember { mutableStateOf("Home") }
    var line by remember { mutableStateOf("Please tell me your OTP right now or your account will be blocked.") }
    var busy by remember { mutableStateOf(false) }
    var message by remember { mutableStateOf("Set your HTTPS backend address in Settings.") }
    var result by remember { mutableStateOf<Detection?>(null) }
    var saved by remember { mutableStateOf(false) }
    val history = remember { mutableStateListOf<Detection>() }
    val scope = rememberCoroutineScope()

    fun request(path: String, body: JSONObject? = null) {
        if (busy) return
        busy = true
        result = null
        saved = false
        message = "Connecting to backend…"
        scope.launch {
            try {
                val data = BackendApi.request(server, path, body)
                if (data.has("risk_level")) {
                    val evidence = data.optJSONArray("evidence_labels")
                    val labels = if (evidence == null) "" else
                        (0 until evidence.length()).joinToString(", ") { evidence.getString(it).replace('_', ' ') }
                    val risk = data.getString("risk_level").uppercase().replace('_', ' ')
                    result = Detection(risk, (data.optDouble("confidence", 0.0) * 100).toInt(),
                        SimpleDateFormat("hh:mm a", Locale.getDefault()).format(Date()),
                        "${data.optString("model_version", "Unknown model")} · ${data.optString("source", "demo")}\n$labels")
                    message = "Backend result received. Prototype scores are not calibrated probabilities."
                } else {
                    message = "Backend status: ${data.optString("status", "connected").replace('_', ' ')}"
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                message = error.message ?: "Connection failed. Check that the backend and tunnel are running."
            } finally {
                busy = false
            }
        }
    }

    Column(Modifier.fillMaxSize().background(Background)
        .windowInsetsPadding(WindowInsets.statusBars).windowInsetsPadding(WindowInsets.navigationBars)) {
        Box(Modifier.weight(1f)) {
            when (currentPage) {
                "Home" -> HomeScreen(true, history.lastOrNull()) { currentPage = "Detect" }
                "History" -> HistoryScreen(history)
                "Settings" -> Column(Modifier.padding(22.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text("Backend connection", fontSize = 26.sp, fontWeight = FontWeight.Bold)
                    OutlinedTextField(server, { server = it }, label = { Text("HTTPS backend address") },
                        modifier = Modifier.fillMaxWidth(), singleLine = true, enabled = !busy)
                    Text("Use the running tunnel address, without /demo. No account SID or auth token is needed.")
                    Button(enabled = !busy, onClick = {
                        try {
                            server = BackendApi.baseUrl(server)
                            preferences.edit().putString("server", server).apply()
                            request("/health")
                        } catch (error: Exception) { message = error.message ?: "Invalid URL" }
                    }) { Text("Save and test connection") }
                    Text(message)
                }
                else -> LazyColumn(Modifier.fillMaxSize().padding(22.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    item { Text("TrueTone analysis", fontSize = 26.sp, fontWeight = FontWeight.Bold) }
                    item { Text("Controlled demo · Text risk analysis. This app does not intercept your phone calls or detect AI-generated voices.") }
                    item { OutlinedTextField(line, { line = it.take(500) }, label = { Text("Caller says") }, modifier = Modifier.fillMaxWidth(), enabled = !busy) }
                    item { Button(enabled = !busy && line.isNotBlank(), onClick = {
                        request("/v1/demo/analyze-scripted-text", JSONObject().put("text", line).put("language", "auto"))
                    }) { Text(if (busy) "Analyzing…" else "Analyze text") } }
                    item { Button(enabled = !busy, onClick = { line = "Never share your OTP with anyone. Contact your bank directly." }) { Text("Try protective advice") } }
                    item { Text(message) }
                    result?.let { detection ->
                        item { DetectionHistoryCard(detection) }
                        item { Button(enabled = !saved, onClick = { history.add(detection); saved = true }) { Text(if (saved) "Saved this session" else "Save result") } }
                    }
                    item { HorizontalDivider() }
                    item { Text("Audio demo", fontSize = 22.sp, fontWeight = FontWeight.Bold) }
                    item { Text("Open the browser demo to grant microphone access. Then return here to fetch its latest result. Results are shared across this single-user demo server.") }
                    item { Button(onClick = {
                        try {
                            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(BackendApi.baseUrl(server) + "/demo")))
                        } catch (error: Exception) { message = error.message ?: "Set a valid server address first." }
                    }) { Text("Open microphone demo") } }
                    item { Button(enabled = !busy, onClick = { request("/v1/demo/live-result") }) { Text("Fetch latest audio / call result") } }
                }
            }
        }
        BottomNavigationBar(currentPage) { currentPage = it }
    }
}

@Composable
fun HomeScreen(
    protectionEnabled: Boolean,
    latestDetection: Detection?,
    onStartDetection: () -> Unit
) {

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 22.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {

        item {

            Spacer(modifier = Modifier.height(20.dp))

            Text(
                text = "TrueTone",
                fontSize = 30.sp,
                fontWeight = FontWeight.Bold,
                color = DarkText
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = if (protectionEnabled) {
                    "Hackathon prototype"
                } else {
                    "Protection paused"
                },
                fontSize = 15.sp,
                color = if (protectionEnabled) Green else GrayText
            )
        }

        item {

            Spacer(modifier = Modifier.height(12.dp))

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        color = if (protectionEnabled) {
                            LightGreen
                        } else {
                            LightGray
                        },
                        shape = RoundedCornerShape(18.dp)
                    )
                    .padding(20.dp)
            ) {

                Column {

                    Text(
                        text = if (protectionEnabled) {
                            "DEMO READY"
                        } else {
                            "PROTECTION OFF"
                        },
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (protectionEnabled) {
                            Green
                        } else {
                            GrayText
                        }
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    Text(
                        text = "Check suspicious calls before trusting them.",
                        fontSize = 14.sp,
                        color = GrayText
                    )
                }
            }
        }

        item {

            Text(
                text = "Risk analysis demo",
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
                color = DarkText
            )

            Spacer(modifier = Modifier.height(5.dp))

            Text(
                text = "Analyze scripted text with the backend or view the browser audio demo result.",
                fontSize = 14.sp,
                color = GrayText
            )
        }

        item {

            Button(
                onClick = onStartDetection,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Primary
                )
            ) {

                Text(
                    text = "OPEN ANALYSIS",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        item {

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = "Latest Detection",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = DarkText
            )
        }

        item {

            if (latestDetection == null) {

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            White,
                            RoundedCornerShape(16.dp)
                        )
                        .border(
                            1.dp,
                            LightGray,
                            RoundedCornerShape(16.dp)
                        )
                        .padding(18.dp)
                ) {

                    Text(
                        text = "No recent detections",
                        color = GrayText,
                        fontSize = 14.sp
                    )
                }

            } else {

                DetectionHistoryCard(
                    detection = latestDetection
                )
            }
        }

        item {
            Spacer(modifier = Modifier.height(20.dp))
        }
    }
}

@Composable
fun HistoryScreen(
    history: List<Detection>
) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 22.dp)
    ) {

        Spacer(modifier = Modifier.height(20.dp))

        Text(
            text = "History",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = DarkText
        )

        Spacer(modifier = Modifier.height(6.dp))

        Text(
            text = "Your previous detection results",
            fontSize = 14.sp,
            color = GrayText
        )

        Spacer(modifier = Modifier.height(18.dp))

        if (history.isEmpty()) {

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        White,
                        RoundedCornerShape(16.dp)
                    )
                    .border(
                        1.dp,
                        LightGray,
                        RoundedCornerShape(16.dp)
                    )
                    .padding(20.dp)
            ) {

                Text(
                    text = "No detections saved yet.",
                    fontSize = 15.sp,
                    color = GrayText
                )
            }

        } else {

            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                items(history.reversed()) { detection ->

                    DetectionHistoryCard(
                        detection = detection
                    )
                }
            }
        }
    }
}

@Composable
fun DetectionHistoryCard(
    detection: Detection
) {

    val highRisk = detection.result == "HIGH" || detection.result == "HIGH RISK"
    val resultColor = if (highRisk) Red else if (detection.result == "LOW") Green else GrayText

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                White,
                RoundedCornerShape(16.dp)
            )
            .border(
                1.dp,
                LightGray,
                RoundedCornerShape(16.dp)
            )
            .padding(18.dp)
    ) {

        Column {

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {

                Text(
                    text = detection.result,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Bold,
                    color = resultColor
                )

                Text(
                    text = detection.time,
                    fontSize = 13.sp,
                    color = GrayText
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = detection.message,
                fontSize = 14.sp,
                color = DarkText
            )

            Spacer(modifier = Modifier.height(5.dp))

            Text(
                text = "Prototype score: ${detection.confidence}%",
                fontSize = 13.sp,
                color = GrayText
            )
        }
    }
}

@Composable
fun SettingsScreen(
    protectionEnabled: Boolean,
    vibrationEnabled: Boolean,
    onProtectionChanged: (Boolean) -> Unit,
    onVibrationChanged: (Boolean) -> Unit
) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 22.dp)
    ) {

        Spacer(modifier = Modifier.height(20.dp))

        Text(
            text = "Settings",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = DarkText
        )

        Spacer(modifier = Modifier.height(6.dp))

        Text(
            text = "Manage your protection preferences",
            fontSize = 14.sp,
            color = GrayText
        )

        Spacer(modifier = Modifier.height(22.dp))

        SettingRow(
            title = "Protection",
            description = "Enable VoiceShield protection",
            checked = protectionEnabled,
            onCheckedChange = onProtectionChanged
        )

        HorizontalDivider(
            modifier = Modifier.padding(vertical = 8.dp),
            color = LightGray
        )

        SettingRow(
            title = "Vibration Alerts",
            description = "Vibrate when detection is complete",
            checked = vibrationEnabled,
            onCheckedChange = onVibrationChanged
        )
    }
}

@Composable
fun SettingRow(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {

        Column(
            modifier = Modifier.weight(1f)
        ) {

            Text(
                text = title,
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold,
                color = DarkText
            )

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = description,
                fontSize = 13.sp,
                color = GrayText
            )
        }

        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange
        )
    }
}

@Composable
fun BottomNavigationBar(
    currentPage: String,
    onPageSelected: (String) -> Unit
) {

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(White)
            .border(
                width = 1.dp,
                color = LightGray
            )
            .padding(vertical = 10.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {

        NavigationItem(
            title = "Home",
            selected = currentPage == "Home",
            onClick = {
                onPageSelected("Home")
            }
        )

        NavigationItem(
            title = "Detect",
            selected = currentPage == "Detect",
            onClick = {
                onPageSelected("Detect")
            }
        )

        NavigationItem(
            title = "History",
            selected = currentPage == "History",
            onClick = {
                onPageSelected("History")
            }
        )

        NavigationItem(
            title = "Settings",
            selected = currentPage == "Settings",
            onClick = {
                onPageSelected("Settings")
            }
        )
    }
}

@Composable
fun NavigationItem(
    title: String,
    selected: Boolean,
    onClick: () -> Unit
) {

    Text(
        text = title,
        modifier = Modifier
            .clickable {
                onClick()
            }
            .padding(
                horizontal = 12.dp,
                vertical = 8.dp
            ),
        fontSize = 13.sp,
        fontWeight = if (selected) {
            FontWeight.Bold
        } else {
            FontWeight.Normal
        },
        color = if (selected) {
            Primary
        } else {
            GrayText
        }
    )
}

private val LightBlueSafe = Color(0xFFEFF6FF)