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
import kotlinx.coroutines.delay
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

    var currentPage by remember { mutableStateOf("Home") }

    var detectionStatus by remember {
        mutableStateOf("ready")
    }

    var protectionEnabled by remember {
        mutableStateOf(true)
    }

    var vibrationEnabled by remember {
        mutableStateOf(true)
    }

    var resultSaved by remember {
        mutableStateOf(false)
    }

    var detectionNumber by remember {
        mutableStateOf(0)
    }

    val history = remember {
        mutableStateListOf<Detection>()
    }

    val context = androidx.compose.ui.platform.LocalContext.current

    fun startDetection() {
        detectionNumber++

        detectionStatus = "analyzing"
        resultSaved = false
        currentPage = "Detect"
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Background)
            .windowInsetsPadding(WindowInsets.statusBars)
            .windowInsetsPadding(WindowInsets.navigationBars)
    ) {

        Box(
            modifier = Modifier.weight(1f)
        ) {

            when (currentPage) {

                "Home" -> {
                    HomeScreen(
                        protectionEnabled = protectionEnabled,
                        latestDetection = history.lastOrNull(),
                        onStartDetection = {
                            startDetection()
                        }
                    )
                }

                "Detect" -> {
                    DetectionScreen(
                        status = detectionStatus,
                        detectionNumber = detectionNumber,
                        resultSaved = resultSaved,
                        vibrationEnabled = vibrationEnabled,

                        onStart = {
                            startDetection()
                        },

                        onSave = {

                            val isHighRisk =
                                detectionNumber % 2 == 1

                            val newDetection = Detection(
                                result = if (isHighRisk) {
                                    "HIGH RISK"
                                } else {
                                    "LOW RISK"
                                },

                                confidence = if (isHighRisk) {
                                    92
                                } else {
                                    96
                                },

                                time = SimpleDateFormat(
                                    "hh:mm a",
                                    Locale.getDefault()
                                ).format(Date()),

                                message = if (isHighRisk) {
                                    "Suspicious OTP request detected"
                                } else {
                                    "No significant spoofing indicators detected"
                                }
                            )

                            history.add(newDetection)
                            resultSaved = true
                        },

                        onViewHistory = {
                            currentPage = "History"
                        },

                        onNewDetection = {
                            startDetection()
                        }
                    )
                }

                "History" -> {
                    HistoryScreen(
                        history = history
                    )
                }

                "Settings" -> {
                    SettingsScreen(
                        protectionEnabled = protectionEnabled,
                        vibrationEnabled = vibrationEnabled,

                        onProtectionChanged = {
                            protectionEnabled = it
                        },

                        onVibrationChanged = {
                            vibrationEnabled = it
                        }
                    )
                }
            }
        }

        BottomNavigationBar(
            currentPage = currentPage,
            onPageSelected = {
                currentPage = it
            }
        )
    }

    LaunchedEffect(detectionStatus) {

        if (detectionStatus == "analyzing") {

            delay(2500)

            detectionStatus = "result"

            if (vibrationEnabled) {
                vibratePhone(context)
            }
        }
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
                text = "VoiceShield",
                fontSize = 30.sp,
                fontWeight = FontWeight.Bold,
                color = DarkText
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = if (protectionEnabled) {
                    "Protection active"
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
                            "PROTECTION ACTIVE"
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
                text = "Live Detection",
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
                color = DarkText
            )

            Spacer(modifier = Modifier.height(5.dp))

            Text(
                text = "Analyze a caller's voice and conversation for suspicious activity.",
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
                    text = "START LIVE DETECTION",
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
fun DetectionScreen(
    status: String,
    detectionNumber: Int,
    resultSaved: Boolean,
    vibrationEnabled: Boolean,
    onStart: () -> Unit,
    onSave: () -> Unit,
    onViewHistory: () -> Unit,
    onNewDetection: () -> Unit
) {

    val isHighRisk = detectionNumber % 2 == 1

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 22.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        item {

            Spacer(modifier = Modifier.height(20.dp))

            Text(
                text = "Live Detection",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = DarkText
            )
        }

        if (status == "ready") {

            item {

                Spacer(modifier = Modifier.height(25.dp))

                Box(
                    modifier = Modifier
                        .size(120.dp)
                        .background(
                            LightBlueSafe,
                            CircleShape
                        ),
                    contentAlignment = Alignment.Center
                ) {

                    Text(
                        text = "MIC",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = Primary
                    )
                }
            }

            item {

                Text(
                    text = "Ready to analyze",
                    fontSize = 21.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkText
                )

                Text(
                    text = "Start detection when you receive a suspicious call.",
                    fontSize = 14.sp,
                    color = GrayText,
                    textAlign = TextAlign.Center
                )
            }

            item {

                Button(
                    onClick = onStart,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(54.dp),
                    shape = RoundedCornerShape(14.dp)
                ) {

                    Text(
                        text = "START DETECTION",
                        fontWeight = FontWeight.Bold
                    )
                }
            }

        } else if (status == "analyzing") {

            item {

                Spacer(modifier = Modifier.height(30.dp))

                Box(
                    modifier = Modifier
                        .size(110.dp)
                        .background(
                            LightBlueSafe,
                            CircleShape
                        ),
                    contentAlignment = Alignment.Center
                ) {

                    Text(
                        text = "•••",
                        fontSize = 28.sp,
                        fontWeight = FontWeight.Bold,
                        color = Primary
                    )
                }
            }

            item {

                Text(
                    text = "Analyzing caller voice...",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkText,
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(5.dp))

                Text(
                    text = "Checking voice patterns and conversation signals",
                    fontSize = 14.sp,
                    color = GrayText,
                    textAlign = TextAlign.Center
                )
            }

        } else {

            item {

                Spacer(modifier = Modifier.height(8.dp))

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            if (isHighRisk) LightRed else LightGreen,
                            RoundedCornerShape(18.dp)
                        )
                        .padding(22.dp)
                ) {

                    Column {

                        Text(
                            text = if (isHighRisk) {
                                "⚠ HIGH RISK"
                            } else {
                                "✓ LOW RISK"
                            },
                            fontSize = 26.sp,
                            fontWeight = FontWeight.Bold,
                            color = if (isHighRisk) Red else Green
                        )

                        Spacer(modifier = Modifier.height(10.dp))

                        Text(
                            text = if (isHighRisk) {
                                "Possible voice impersonation / suspicious request detected"
                            } else {
                                "No significant spoofing indicators detected"
                            },
                            fontSize = 15.sp,
                            color = DarkText
                        )

                        Spacer(modifier = Modifier.height(14.dp))

                        Text(
                            text = if (isHighRisk) {
                                "92% risk confidence"
                            } else {
                                "96% confidence"
                            },
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = DarkText
                        )
                    }
                }
            }

            item {

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

                        Text(
                            text = "Caller conversation",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            color = DarkText
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = if (isHighRisk) {
                                "\"I'm from your bank. Please tell me the OTP.\""
                            } else {
                                "\"Hello, I'm calling regarding your appointment.\""
                            },
                            fontSize = 15.sp,
                            color = GrayText
                        )
                    }
                }
            }

            item {

                if (!resultSaved) {

                    Button(
                        onClick = onSave,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {

                        Text(
                            text = "SAVE TO HISTORY",
                            fontWeight = FontWeight.Bold
                        )
                    }

                } else {

                    Button(
                        onClick = onViewHistory,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {

                        Text(
                            text = "VIEW HISTORY",
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            item {

                Button(
                    onClick = onNewDetection,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = LightGray,
                        contentColor = DarkText
                    )
                ) {

                    Text(
                        text = "NEW DETECTION",
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            item {

                Text(
                    text = if (vibrationEnabled) {
                        "Alert vibration enabled"
                    } else {
                        "Alert vibration disabled"
                    },
                    fontSize = 13.sp,
                    color = GrayText
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

    val highRisk = detection.result == "HIGH RISK"

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
                    color = if (highRisk) Red else Green
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
                text = "${detection.confidence}% confidence",
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