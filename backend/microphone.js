// Consented browser microphone demo. PCM goes over WSS; no audio is saved here.
let stream = null, context = null, source = null, processor = null, socket = null;
let autoStop = null, pollTimer = null, samplesSent = 0, peakSeen = 0, active = false;
const startButton = document.getElementById('micStart');
const stopButton = document.getElementById('micStop');
const indicator = document.getElementById('micIndicator');
const liveText = document.getElementById('live');
const micStats = document.getElementById('micStats');
const callStatus = document.getElementById('callStatus');

async function pollIncomingCall() {
  try {
    const response = await fetch('/v1/demo/live-result', {cache: 'no-store'});
    const result = await response.json();
    if (result.source === 'twilio_gather') {
      const labels = result.evidence_labels?.length ? ' · ' + result.evidence_labels.join(', ') : '';
      callStatus.textContent = 'Phone speech (Twilio recognition): ' +
        result.risk_level.replaceAll('_', ' ').toUpperCase() + labels +
        ' · ' + Math.round(result.confidence * 100) + '% · text prototype';
      return;
    }
    if (result.source !== 'twilio_media') return;
    const line = result.line_type;
    const type = line === 'fixedVoip' || line === 'nonFixedVoip' ? 'VoIP'
      : line === 'unknown' ? 'Unknown' : 'Not classified as VoIP';
    const risk = result.status === 'analyzed'
      ? ' · Speech risk: ' + result.risk_level.replaceAll('_', ' ').toUpperCase()
      : ' · Waiting for speech analysis';
    callStatus.textContent = 'Caller line: ' + type + ' (' + line + ')' + risk;
  } catch (_) { /* Retry on the next poll. */ }
}
setInterval(pollIncomingCall, 3000);

function show(message, state = '') {
  liveText.textContent = message;
  indicator.className = state;
}

async function pollResult() {
  try {
    const response = await fetch('/v1/demo/live-result', {cache: 'no-store'});
    const result = await response.json();
    if (result.status === 'processing') {
      show('Audio received. Whisper is analyzing it; please wait up to 60 seconds.', 'processing');
    } else if (result.status === 'analyzed') {
      const labels = result.evidence_labels.length ? ' · ' + result.evidence_labels.join(', ') : '';
      const diagnostic = result.risk_level === 'insufficient_evidence'
        ? (result.language === 'unknown' ? ' · Whisper returned no usable words' : ' · Speech recognized, but no risk phrase matched')
        : '';
      show('Live audio: ' + result.risk_level.replaceAll('_', ' ').toUpperCase() + labels +
           ' · ' + Math.round(result.confidence * 100) + '% · ' + result.model_version + diagnostic,
           active ? 'recording' : '');
      if (!active && result.risk_level !== 'insufficient_evidence') stopPolling();
    } else if (result.status === 'analysis_unavailable') {
      show('Audio reached the backend, but speech analysis is unavailable. Try again.');
      if (!active) stopPolling();
    }
  } catch (_) { /* The next poll can recover from a brief network interruption. */ }
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function stopMic() {
  if (autoStop) clearTimeout(autoStop);
  autoStop = null;
  active = false;
  if (processor) { processor.onaudioprocess = null; processor.disconnect(); processor = null; }
  if (source) { source.disconnect(); source = null; }
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  if (context) { await context.close(); context = null; }
  if (socket) { socket.close(); socket = null; }
  startButton.disabled = false;
  stopButton.disabled = true;
  show(samplesSent >= 64000
    ? 'Microphone stopped. Waiting for the analyzed result…'
    : 'Microphone stopped before four seconds of audio. Try again.', 'processing');
  if (samplesSent < 64000) stopPolling();
}

async function startMic() {
  if (startButton.disabled) return;
  startButton.disabled = true;
  samplesSent = 0;
  peakSeen = 0;
  micStats.textContent = 'Audio sent: 0 seconds · Input peak: 0%';
  show('Requesting microphone permission…', 'processing');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone access is unavailable in this browser');
    stream = await navigator.mediaDevices.getUserMedia({audio: true});
    show('Microphone allowed. Connecting to backend…', 'processing');
    socket = new WebSocket(location.origin.replace(/^http/, 'ws') + '/v1/demo/microphone');
    socket.binaryType = 'arraybuffer';
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = () => reject(new Error('Could not connect to audio stream'));
      socket.onclose = event => reject(new Error('Audio stream closed: code ' + event.code));
    });
    socket.onclose = event => {
      if (active) {
        stopMic().then(() => show('Audio stream disconnected: code ' + event.code + '. Tap Start microphone to retry.'));
      }
    };
    context = new (window.AudioContext || window.webkitAudioContext)();
    await context.resume();
    source = context.createMediaStreamSource(stream);
    processor = context.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = event => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      for (let i = 0; i < input.length; i++) peakSeen = Math.max(peakSeen, Math.abs(input[i]));
      const ratio = context.sampleRate / 16000;
      const pcm = new Int16Array(Math.floor(input.length / ratio));
      for (let i = 0; i < pcm.length; i++) {
        const value = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
        pcm[i] = value < 0 ? value * 32768 : value * 32767;
      }
      socket.send(pcm.buffer);
      samplesSent += pcm.length;
      if (samplesSent % 16000 < pcm.length) {
        micStats.textContent = 'Audio sent: ' + (samplesSent / 16000).toFixed(1) +
          ' seconds · Input peak: ' + Math.round(peakSeen * 100) + '%';
        show('Recording now · ' + Math.floor(samplesSent / 16000) + ' seconds sent. Repeat “मुझे अपना ओटीपी बताओ”.', 'recording');
      }
    };
    source.connect(processor);
    processor.connect(context.destination);
    active = true;
    stopButton.disabled = false;
    show('Recording now. Repeat “मुझे अपना ओटीपी बताओ” for 10 seconds, then tap Stop.', 'recording');
    pollTimer = setInterval(pollResult, 2000);
    autoStop = setTimeout(stopMic, 25000);
  } catch (error) {
    const detail = error?.message || error?.name || 'unknown browser error';
    await stopMic();
    show('Microphone failed: ' + detail + '. Close other demo microphone tabs and retry.');
  }
}

startButton.onclick = startMic;
stopButton.onclick = stopMic;
