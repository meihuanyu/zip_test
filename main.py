from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sherpa_onnx
import numpy as np
import wave
import io
import json
import re
import time
from typing import Optional, Tuple, List, Dict, Union
from collections import defaultdict
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from streaming_aligner import StreamingAligner
import sentencepiece as spm
import torch
from example_onnx_inference import SimpleOnnxModel, extract_fbank
from ctc_align_beam import StreamingCTCAligner, AlignConfig

app = FastAPI()

# 模型配置路径（请根据实际情况修改）
ENCODER_PATH = "data/ctc-epoch-30-avg-1-chunk-16-left-128.onnx"
TOKENS_PATH = "tokens.txt"
SAMPLE_RATE = 16000

# 新的推理模型配置
ONNX_MODEL_PATH = "data/ctc-epoch-30-avg-1-chunk-16-left-128.onnx"  # 根据实际情况修改
BPE_MODEL_PATH = "data/lang_bpe_500/bpe.model"  # 根据实际情况修改

recognizer = None
bpe_model: Optional[spm.SentencePieceProcessor] = None

# 存储每个WebSocket连接的SimpleOnnxModel实例和对齐器
phoneme_models: Dict[str, SimpleOnnxModel] = {}
phoneme_aligners: Dict[str, StreamingCTCAligner] = {}
phoneme_bpe_models: Dict[str, spm.SentencePieceProcessor] = {}

async def safe_send_json(websocket: WebSocket, data: dict):
    """安全地发送JSON数据，如果连接已关闭则忽略异常"""
    try:
        await websocket.send_json(data)
    except (ConnectionClosedOK, ConnectionClosedError, WebSocketDisconnect):
        # 连接已关闭，忽略错误
        pass

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

def parse_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """从PCM字节流解析音频数据（16位小端序）"""
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio

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
        result = recognizer.get_result_all(stream)
        
        return {
            "text": result.text if hasattr(result, 'text') else str(result),
            "tokens": result.tokens if hasattr(result, 'tokens') else [],
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

class AlignmentRequest(BaseModel):
    text: str
    
@app.websocket("/api/alignment/ws")
async def websocket_alignment(websocket: WebSocket):
    """WebSocket端点：实时接收音频流并进行对齐"""
    await websocket.accept()
    stream_id = str(id(websocket))
    
    if recognizer is None:
        print("[WebSocket] 错误: 模型未加载")
        await safe_send_json(websocket, {"error": "模型未加载"})
        await websocket.close()
        return
    
    try:
        # 创建新的stream
        stream = recognizer.create_stream()
        streams[stream_id] = stream
        
        # 接收参考文本
        init_data = await websocket.receive_json()
        reference_text = init_data.get("text", "")
        
        # 初始化对齐器
        aligner = StreamingAligner(reference_text, TOKENS_PATH)
        processed_token_count = 0
        
        await safe_send_json(websocket, {"status": "ready"})
        
        # 累积音频缓冲区
        audio_buffer = np.array([], dtype=np.float32)
        min_chunk_size = int(SAMPLE_RATE * 0.4)  # 保持 0.47s 缓冲
        last_sent_index = -1
        total_samples = 0
        
        while True:
            # 接收音频数据
            data = await websocket.receive()
            
            if "bytes" in data:
                # 接收PCM音频数据
                audio_bytes = data["bytes"]
                audio_chunk = parse_audio_from_bytes(audio_bytes)
                
                # 累积音频数据
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                # 当累积足够的数据时再处理
                if len(audio_buffer) >= min_chunk_size:
                    current_buffer_len = len(audio_buffer)
                    t_start_infer = time.perf_counter()
                    
                    # 处理累积的音频数据
                    stream.accept_waveform(SAMPLE_RATE, audio_buffer.astype(np.float32))
                    
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    
                    t_end_infer = time.perf_counter()
                    
                    total_samples += len(audio_buffer)
                    current_time = total_samples / SAMPLE_RATE
                    
                    # 清空缓冲区
                    audio_buffer = np.array([], dtype=np.float32)
                    
                    # 获取当前识别结果
                    t_start_align = time.perf_counter()
                    result = recognizer.get_result_all(stream)
                    # 处理新 Tokens
                    if hasattr(result, 'tokens'):
                        tokens = result.tokens
                        timestamps = result.timestamps if hasattr(result, 'timestamps') else []
                        
                        if len(tokens) > processed_token_count:
                            new_tokens = tokens[processed_token_count:]
                            print(f"[DEBUG] New tokens ({len(new_tokens)}): {new_tokens}")
                            
                            # 发送 new_tokens 用于调试
                            await safe_send_json(websocket, {
                                "new_tokens": new_tokens,
                                "token_count": len(new_tokens),
                                "total_tokens": len(tokens),
                                "current_time": current_time
                            })
                            
                            for i, token in enumerate(new_tokens):
                                idx = processed_token_count + i
                                # 获取对应的时间戳，如果没有则用当前时间估算
                                t_start = timestamps[idx] if idx < len(timestamps) else current_time
                                t_end = -1.0 # aligner 会估算
                                
                                event = aligner.process_token(token, t_start, t_end)
                                if event:
                                    # 只有索引向前推进时才发送
                                    if event['index'] != last_sent_index:
                                        print(f"[WebSocket] 匹配: 索引={event['index']}, 文本={event['text']}, 时间={event['start']:.2f}s")
                                        await safe_send_json(websocket, {
                                            "index": event['index'],
                                            "current_time": event['start'],
                                            "recognized_text": event['text']
                                        })
                                        last_sent_index = event['index']
                            
                            processed_token_count = len(tokens)
                    
                    t_end_align = time.perf_counter()
                    
                    # 打印性能统计
                    audio_duration_ms = (current_buffer_len / SAMPLE_RATE) * 1000
                    infer_ms = (t_end_infer - t_start_infer) * 1000
                    align_ms = (t_end_align - t_start_align) * 1000
                    rtf = infer_ms / audio_duration_ms if audio_duration_ms > 0 else 0
                    # print(f"[Perf] Audio: {audio_duration_ms:.1f}ms | Infer: {infer_ms:.1f}ms (x{decode_count}) | Align: {align_ms:.1f}ms | RTF: {rtf:.2f}")
            
            elif "text" in data:
                # 接收文本消息（如停止信号）
                msg = json.loads(data["text"])
                if msg.get("action") == "stop":
                    # 处理缓冲区中剩余的音频数据
                    if len(audio_buffer) > 0:
                        stream.accept_waveform(SAMPLE_RATE, audio_buffer.astype(np.float32))
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                    
                    stream.input_finished()
                    print(f"[WebSocket] Stream Stop.")
                    break
    
    except WebSocketDisconnect:
        print(f"[WebSocket] 连接断开: {stream_id}")
    except Exception as e:
        print(f"[WebSocket] 错误: {e}")
        import traceback
        traceback.print_exc()
        await safe_send_json(websocket, {"error": str(e)})
    finally:
        if stream_id in streams:
            del streams[stream_id]
            print(f"[WebSocket] 清理stream: {stream_id}")

@app.websocket("/api/phoneme/ws")
async def websocket_phoneme(websocket: WebSocket):
    """WebSocket端点：使用SimpleOnnxModel获取CTC log_probs，使用StreamingCTCAligner进行对齐"""
    await websocket.accept()
    stream_id = str(id(websocket))
    
    try:
        # 接收参考文本
        init_data = await websocket.receive_json()
        reference_text = init_data.get("text", "")
        
        if not reference_text:
            await safe_send_json(websocket, {"error": "参考文本不能为空"})
            await websocket.close()
            return
        
        # 加载BPE模型
        try:
            sp = spm.SentencePieceProcessor()
            sp.load(BPE_MODEL_PATH)
            phoneme_bpe_models[stream_id] = sp
        except Exception as e:
            await safe_send_json(websocket, {"error": f"BPE模型加载失败: {str(e)}"})
            await websocket.close()
            return
        
        # 将文本编码为token IDs
        target_tokens_list = sp.encode(reference_text)
        if len(target_tokens_list) == 0:
            await safe_send_json(websocket, {"error": "文本编码后长度为0"})
            await websocket.close()
            return
        
        target_tokens = torch.tensor(target_tokens_list, dtype=torch.long)
        blank_id = sp.piece_to_id("<blk>")
        
        # 初始化SimpleOnnxModel
        try:
            model = SimpleOnnxModel(ONNX_MODEL_PATH)
            phoneme_models[stream_id] = model
        except Exception as e:
            await safe_send_json(websocket, {"error": f"ONNX模型加载失败: {str(e)}"})
            await websocket.close()
            return
        
        # 初始化对齐器
        align_config = AlignConfig()
        aligner = StreamingCTCAligner(
            target_tokens=target_tokens,
            blank_id=blank_id,
            config=align_config,
        )
        phoneme_aligners[stream_id] = aligner
        
        await safe_send_json(websocket, {"status": "ready"})
        
        # 音频缓冲区
        audio_buffer = np.array([], dtype=np.float32)
        min_chunk_size = int(SAMPLE_RATE * 0.4)  # 0.4秒缓冲
        last_token_index = -1
        
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                # 接收PCM音频数据
                audio_bytes = data["bytes"]
                audio_chunk = parse_audio_from_bytes(audio_bytes)
                
                # 累积音频数据
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                # 当累积足够的数据时再处理
                if len(audio_buffer) >= min_chunk_size:
                    # 提取fbank特征
                    audio_tensor = torch.from_numpy(audio_buffer).float()
                    features = extract_fbank(audio_tensor, SAMPLE_RATE)  # (T, 80)
                    
                    # 将特征分割成chunk进行处理
                    T = model.T  # 45
                    offset = model.decode_chunk_len  # 32
                    num_frames = features.shape[0]
                    
                    # 计算可以处理的帧数（保留一些帧用于下次处理）
                    processed_frames = 0
                    start_idx = 0
                    while start_idx + T <= num_frames:
                        chunk_features = features[start_idx:start_idx + T]  # (T, 80)
                        chunk_features = chunk_features[np.newaxis, :, :]  # (1, T, 80)
                        
                        # 推理获取CTC log_probs
                        log_probs = model(chunk_features)  # (1, chunk_size, vocab_size)

                        def print_ctc(ctc_output, blank_id, sp):
                            # 调试：显示有意义的帧（非blank概率较高的帧）
                            log_probs = ctc_output  # ctc_output 已经是 log_softmax 输出
                            probs = np.exp(log_probs)
                            T = probs.shape[1]
                            blank_prob = probs[0, :, blank_id]
                            
                            # 统计信息
                            non_blank_frames = np.sum(blank_prob < 0.5)
                            
                            # 只显示非blank概率较高的帧
                            print("\n有意义的帧 (非blank概率 > 0.1):")
                            for t in range(T):
                                blank_p = float(blank_prob[t])
                                if blank_p < 0.9:  # 只显示非blank概率 > 0.1 的帧
                                    frame_probs = probs[0, t, :]
                                    top_k = 3
                                    top_indices = np.argpartition(frame_probs, -top_k)[-top_k:]
                                    top_indices = top_indices[np.argsort(frame_probs[top_indices])][::-1]
                                    top_probs = frame_probs[top_indices]
                                    
                                    items = []
                                    for idx, prob in zip(top_indices, top_probs):
                                        p = float(prob)
                                        token_name = sp.id_to_piece(int(idx))
                                        items.append(f"{token_name}:{p:.2f}")
                                    
                                    line = ", ".join(items)
                                    print(f"  frame {t:3d}: {line}")
                        
                        print_ctc(log_probs, blank_id, sp)
                        # 逐帧调用对齐器
                        batch_size, T_out, vocab_size = log_probs.shape
                        for t in range(T_out):
                            log_probs_t = torch.from_numpy(log_probs[0, t, :]).float()
                            current_token_index = aligner.step(log_probs_t)
                            
                            # 只有当token_index变化时才发送
                            if current_token_index != last_token_index and current_token_index < len(target_tokens_list):
                                last_token_index = current_token_index
                                current_phone = sp.id_to_piece(target_tokens_list[current_token_index])
                                aligned_text = sp.decode(target_tokens_list[:current_token_index + 1])
                                
                                await safe_send_json(websocket, {
                                    "token_index": int(current_token_index),
                                    "current_phone": current_phone,
                                    "aligned_text": aligned_text,
                                    "status": "processing"
                                })
                        
                        processed_frames += offset
                        start_idx += offset
                    
                    # 保留未处理的音频数据（大约保留最后0.1秒）
                    keep_samples = int(SAMPLE_RATE * 0.1)
                    if len(audio_buffer) > keep_samples:
                        audio_buffer = audio_buffer[-keep_samples:]
                    # 如果处理了所有数据，清空缓冲区
                    if processed_frames >= num_frames - T:
                        audio_buffer = np.array([], dtype=np.float32)
            
            elif "text" in data:
                # 接收文本消息（如停止信号）
                msg = json.loads(data["text"])
                if msg.get("action") == "stop":
                    print(f"[WebSocket] Phoneme Stream Stop: {stream_id}")
                    break
    
    except WebSocketDisconnect:
        print(f"[WebSocket] Phoneme连接断开: {stream_id}")
    except Exception as e:
        print(f"[WebSocket] Phoneme错误: {e}")
        import traceback
        traceback.print_exc()
        await safe_send_json(websocket, {"error": str(e)})
    finally:
        # 清理资源
        if stream_id in phoneme_models:
            del phoneme_models[stream_id]
        if stream_id in phoneme_aligners:
            del phoneme_aligners[stream_id]
        if stream_id in phoneme_bpe_models:
            del phoneme_bpe_models[stream_id]
        print(f"[WebSocket] 清理phoneme实例: {stream_id}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回 Web UI"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/alignment", response_class=HTMLResponse)
async def alignment_page():
    """返回对齐页面"""
    with open("static/alignment.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/debug-tokens", response_class=HTMLResponse)
async def debug_tokens_page():
    """返回 new_tokens 调试页面"""
    with open("static/debug_tokens.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/simple", response_class=HTMLResponse)
async def simple_page():
    """返回简单推理测试页面"""
    with open("static/simple.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/phoneme", response_class=HTMLResponse)
async def phoneme_page():
    """返回音素预测页面"""
    with open("static/phoneme.html", "r", encoding="utf-8") as f:
        return f.read()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8068)
