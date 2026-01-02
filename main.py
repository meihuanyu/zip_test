from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sherpa_onnx
import numpy as np
import wave
import io
from typing import Optional, Tuple

app = FastAPI()

# 模型配置路径（请根据实际情况修改）
ENCODER_PATH = "ctc-epoch-18-avg-1-chunk-16-left-128.onnx"
TOKENS_PATH = "tokens.txt"
SAMPLE_RATE = 16000

recognizer = None

def load_model():
    """加载 sherpa-onnx CTC 模型和 tokens"""
    global recognizer
    try:
        # 使用 OnlineRecognizer.from_zipformer2_ctc 加载流式 CTC 模型
        recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=TOKENS_PATH,
            model=ENCODER_PATH,
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method='greedy_search',
            debug=False,
        )
        print(f"CTC 模型和 tokens 加载成功（流式模式）")
        print(f"  - Model: {ENCODER_PATH}")
        print(f"  - Tokens: {TOKENS_PATH}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        recognizer = None

@app.on_event("startup")
async def startup_event():
    load_model()

def read_wav_bytes(audio_data: bytes) -> Tuple[np.ndarray, int]:
    """从字节流读取 WAV 音频数据"""
    with wave.open(io.BytesIO(audio_data)) as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_samples = wav_file.getnframes()
        
        # 读取音频数据
        audio_bytes = wav_file.readframes(n_samples)
        
        # 转换为 numpy 数组
        dtype = np.int16 if wav_file.getsampwidth() == 2 else np.int8
        audio = np.frombuffer(audio_bytes, dtype=dtype)
        
        # 转换为 float32，归一化到 [-1, 1]
        if dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32) / 128.0
        
        # 如果是立体声，转换为单声道
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        return audio, sample_rate

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """简单的线性重采样"""
    if orig_sr == target_sr:
        return audio
    
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio)

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """使用 sherpa-onnx 流式转录音频文件"""
    if recognizer is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 读取上传的音频文件
        audio_data = await file.read()
        
        # 读取 WAV 音频
        audio, sample_rate = read_wav_bytes(audio_data)
        
        # 重采样到目标采样率
        if sample_rate != SAMPLE_RATE:
            audio = resample_audio(audio, sample_rate, SAMPLE_RATE)
        
        # 使用 OnlineRecognizer 进行流式识别
        stream = recognizer.create_stream()
        
        # 分块处理音频（每次处理 0.5 秒，确保有足够的数据）
        chunk_size = SAMPLE_RATE // 2  # 0.5 秒
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size].astype(np.float32)
            stream.accept_waveform(SAMPLE_RATE, chunk)
            recognizer.decode_stream(stream)
        
        # 输入完成，获取最终结果
        stream.input_finished()
        result = recognizer.get_result(stream)
        
        return {
            "text": result.text if hasattr(result, 'text') else str(result),
            "tokens": result.tokens if hasattr(result, 'tokens') else [],
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回 Web UI"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
