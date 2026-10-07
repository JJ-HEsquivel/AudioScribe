// AudioScribe server (run: npm start). Serves the page and transcribes uploaded audio.
const http = require("http");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { exec } = require("child_process");
const { transcribeFile } = require("./transcriber");

const PORT = process.env.PORT || 3000;
const MODELS = new Set([
  "onnx-community/whisper-base",
  "onnx-community/whisper-small",
  "onnx-community/whisper-large-v3-turbo",
]);

// One transcription at a time: the model uses all the CPU.
let busy = false;

async function handleTranscribe(req, res, url) {
  const model = MODELS.has(url.searchParams.get("model"))
    ? url.searchParams.get("model")
    : "onnx-community/whisper-small";

  if (busy) {
    res.writeHead(409, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Ya hay una transcripción en curso. Espera a que termine." }));
    return;
  }
  busy = true;

  const tmp = path.join(os.tmpdir(), `audioscribe-${Date.now()}.mp4`);
  res.writeHead(200, { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache" });
  const emit = (event) => res.write(JSON.stringify(event) + "\n");

  try {
    await new Promise((resolve, reject) => {
      const out = fs.createWriteStream(tmp);
      req.pipe(out);
      out.on("finish", resolve);
      out.on("error", reject);
      req.on("error", reject);
    });
    const result = await transcribeFile(tmp, model, emit);
    emit({ type: "done", result });
  } catch (err) {
    console.error(err);
    emit({ type: "error", message: err?.message || String(err) });
  } finally {
    busy = false;
    res.end();
    fs.unlink(tmp, () => {});
  }
}

http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "POST" && url.pathname === "/transcribe") {
    handleTranscribe(req, res, url);
    return;
  }

  if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
    fs.readFile(path.join(__dirname, "index.html"), (err, data) => {
      if (err) return res.writeHead(500).end();
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
      res.end(data);
    });
    return;
  }

  res.writeHead(404).end("Not found");
}).listen(PORT, () => {
  const link = `http://localhost:${PORT}`;
  console.log(`AudioScribe listo en ${link}  (Ctrl + C para detener)`);
  if (process.platform === "win32" && !process.env.NO_OPEN) exec(`start "" "${link}"`);
});
