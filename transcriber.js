// Whisper transcription in Node: ffmpeg decodes the file, transformers.js runs the model on CPU.
const { spawn } = require("child_process");
const path = require("path");
const ffmpegPath = require("ffmpeg-static");

const SAMPLE_RATE = 16000;
let lib = null;
let transcriber = null;
let loadedModel = null;

// Decode any audio/video file to 16 kHz mono float32 (what Whisper expects).
function decodeAudio(filePath) {
  return new Promise((resolve, reject) => {
    const ff = spawn(ffmpegPath, [
      "-v", "error", "-i", filePath,
      "-vn", "-ac", "1", "-ar", String(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ]);
    const chunks = [];
    let stderr = "";
    ff.stdout.on("data", (d) => chunks.push(d));
    ff.stderr.on("data", (d) => (stderr += d));
    ff.on("error", reject);
    ff.on("close", (code) => {
      if (code !== 0) return reject(new Error(`No se pudo leer el audio: ${stderr.trim()}`));
      const buf = Buffer.concat(chunks);
      const copy = new Uint8Array(buf.length - (buf.length % 4));
      copy.set(buf.subarray(0, copy.length));
      resolve(new Float32Array(copy.buffer));
    });
  });
}

async function getTranscriber(model, emit) {
  if (!lib) {
    lib = await import("@huggingface/transformers");
    lib.env.cacheDir = path.join(__dirname, "models");
  }
  if (transcriber && loadedModel === model) return transcriber;
  if (transcriber) await transcriber.dispose?.();
  transcriber = null;
  emit({ type: "status", message: "Cargando modelo…" });
  const load = () => lib.pipeline("automatic-speech-recognition", model, {
    device: "cpu",
    dtype: { encoder_model: "fp32", decoder_model_merged: "q8" },
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        emit({ type: "download", file: p.file, loaded: p.loaded, total: p.total });
      }
    },
  });
  try {
    transcriber = await load();
  } catch (err) {
    // Right after the first download, antivirus can briefly lock the model files.
    await new Promise((r) => setTimeout(r, 3000));
    transcriber = await load();
  }
  loadedModel = model;
  return transcriber;
}

async function transcribeFile(filePath, model, emit) {
  emit({ type: "status", message: "Leyendo audio…" });
  const audio = await decodeAudio(filePath);
  const seconds = audio.length / SAMPLE_RATE;
  emit({ type: "audio", seconds });

  const asr = await getTranscriber(model, emit);

  // Same windowing the pipeline uses: 30 s windows advancing 20 s each time.
  const win = 30 * SAMPLE_RATE, jump = 20 * SAMPLE_RATE;
  const total = audio.length <= win ? 1 : 1 + Math.ceil((audio.length - win) / jump);
  let done = 0;
  emit({ type: "progress", done, total });

  const streamer = new lib.WhisperTextStreamer(asr.tokenizer, {
    skip_prompt: true,
    callback_function: (text) => emit({ type: "partial", text }),
    on_finalize: () => emit({ type: "progress", done: Math.min(++done, total), total }),
  });

  const result = await asr(audio, {
    language: "spanish",
    task: "transcribe",
    chunk_length_s: 30,
    stride_length_s: 5,
    return_timestamps: true,
    streamer,
  });
  return result;
}

module.exports = { transcribeFile };
