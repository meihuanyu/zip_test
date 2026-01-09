#!/usr/bin/env python3
"""
最简单的 ONNX 流式 CTC 模型调用示例
"""
import argparse
import onnxruntime as ort
import numpy as np
import torch
import torchaudio

try:
    import kaldifeat
    HAS_KALDIFEAT = True
except ImportError:
    HAS_KALDIFEAT = False
    print("警告: 未安装 kaldifeat，将使用 torchaudio 的 fbank")

class SimpleOnnxModel:
    def __init__(self, model_path: str):
        # 创建 ONNX Runtime 会话
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        
        # 从模型元数据获取配置信息
        meta = self.session.get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(meta["decode_chunk_len"])  # 32
        self.T = int(meta["T"])  # 45 (chunk_size*2 + pad_length)
        
        # 初始化状态
        self._init_states()
    
    def _init_states(self, batch_size: int = 1):
        """初始化流式推理的状态"""
        meta = self.session.get_modelmeta().custom_metadata_map
        
        # 解析元数据
        num_encoder_layers = list(map(int, meta["num_encoder_layers"].split(",")))
        encoder_dims = list(map(int, meta["encoder_dims"].split(",")))
        cnn_module_kernels = list(map(int, meta["cnn_module_kernels"].split(",")))
        left_context_len = list(map(int, meta["left_context_len"].split(",")))
        query_head_dims = list(map(int, meta["query_head_dims"].split(",")))
        value_head_dims = list(map(int, meta["value_head_dims"].split(",")))
        num_heads = list(map(int, meta["num_heads"].split(",")))
        
        self.states = []
        
        # 为每个编码器层创建状态
        for i in range(len(num_encoder_layers)):
            num_layers = num_encoder_layers[i]
            key_dim = query_head_dims[i] * num_heads[i]
            embed_dim = encoder_dims[i]
            nonlin_attn_head_dim = 3 * embed_dim // 4
            value_dim = value_head_dims[i] * num_heads[i]
            conv_left_pad = cnn_module_kernels[i] // 2
            
            for layer in range(num_layers):
                # 6个状态张量：key, nonlin_attn, val1, val2, conv1, conv2
                self.states.append(np.zeros((left_context_len[i], batch_size, key_dim), dtype=np.float32))
                self.states.append(np.zeros((1, batch_size, left_context_len[i], nonlin_attn_head_dim), dtype=np.float32))
                self.states.append(np.zeros((left_context_len[i], batch_size, value_dim), dtype=np.float32))
                self.states.append(np.zeros((left_context_len[i], batch_size, value_dim), dtype=np.float32))
                self.states.append(np.zeros((batch_size, embed_dim, conv_left_pad), dtype=np.float32))
                self.states.append(np.zeros((batch_size, embed_dim, conv_left_pad), dtype=np.float32))
        
        # embed_states: (batch_size, 128, 3, 19)
        self.states.append(np.zeros((batch_size, 128, 3, 19), dtype=np.float32))
        # processed_lens: (batch_size,)
        self.states.append(np.zeros(batch_size, dtype=np.int64))
    
    def _build_input_dict(self, x: np.ndarray) -> dict:
        """构建模型输入字典"""
        inputs = {"x": x}
        
        # 按照导出时的顺序构建状态输入
        # 状态顺序：encoder layers (每个layer 6个状态) -> embed_states -> processed_lens
        state_idx = 0
        num_encoder_states = len(self.states) - 2  # 减去最后两个：embed_states 和 processed_lens
        num_encoders = num_encoder_states // 6
        
        for i in range(num_encoders):
            # 每个 encoder layer 有 6 个状态
            inputs[f"cached_key_{i}"] = self.states[state_idx]
            state_idx += 1
            inputs[f"cached_nonlin_attn_{i}"] = self.states[state_idx]
            state_idx += 1
            inputs[f"cached_val1_{i}"] = self.states[state_idx]
            state_idx += 1
            inputs[f"cached_val2_{i}"] = self.states[state_idx]
            state_idx += 1
            inputs[f"cached_conv1_{i}"] = self.states[state_idx]
            state_idx += 1
            inputs[f"cached_conv2_{i}"] = self.states[state_idx]
            state_idx += 1
        
        # 最后两个状态
        inputs["embed_states"] = self.states[-2]
        inputs["processed_lens"] = self.states[-1]
        
        return inputs
    
    def _update_states_from_output(self, outputs: list):
        """从输出更新状态"""
        # 第一个输出是 log_probs，其余是新状态
        num_encoder_states = len(self.states) - 2
        num_encoders = num_encoder_states // 6
        
        state_idx = 0
        for i in range(num_encoders):
            self.states[state_idx] = outputs[1 + i * 6]  # new_cached_key_{i}
            self.states[state_idx + 1] = outputs[1 + i * 6 + 1]  # new_cached_nonlin_attn_{i}
            self.states[state_idx + 2] = outputs[1 + i * 6 + 2]  # new_cached_val1_{i}
            self.states[state_idx + 3] = outputs[1 + i * 6 + 3]  # new_cached_val2_{i}
            self.states[state_idx + 4] = outputs[1 + i * 6 + 4]  # new_cached_conv1_{i}
            self.states[state_idx + 5] = outputs[1 + i * 6 + 5]  # new_cached_conv2_{i}
            state_idx += 6
        
        # 最后两个状态
        self.states[-2] = outputs[-2]  # new_embed_states
        self.states[-1] = outputs[-1]  # new_processed_lens
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        执行流式推理
        
        Args:
            x: 输入特征，shape 为 (batch_size, T, 80)
               T 应该是 45 (decode_chunk_len*2 + pad_length)
        
        Returns:
            log_probs: shape (batch_size, T_out, vocab_size) 的 log 概率
        """
        # 构建输入字典
        inputs = self._build_input_dict(x)
        
        # 获取输出名称（按顺序）
        output_names = [output.name for output in self.session.get_outputs()]
        
        # 运行推理
        outputs = self.session.run(output_names, inputs)
        
        # 更新状态
        self._update_states_from_output(outputs)
        
        # 返回 log_probs（第一个输出）
        return outputs[0]

    def reset_states(self):
        """重置状态，用于处理新的音频"""
        self._init_states()


def load_audio(audio_file: str, sample_rate: int = 16000) -> torch.Tensor:
    """加载音频文件"""
    wave, sr = torchaudio.load(audio_file)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        wave = resampler(wave)
    # 使用第一个通道，转换为单声道
    if wave.shape[0] > 1:
        wave = wave[0:1]
    return wave.squeeze(0).contiguous()  # (T,)


def extract_fbank(wave: torch.Tensor, sample_rate: int = 16000, num_mel_bins: int = 80):
    """提取 fbank 特征"""
    if HAS_KALDIFEAT:
        opts = kaldifeat.FbankOptions()
        opts.device = "cpu"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sample_rate
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.high_freq = -400
        
        fbank = kaldifeat.Fbank(opts)
        feature = fbank(wave)  # (T, 80)
        return feature.numpy()
    else:
        # 使用 torchaudio 作为备选
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=num_mel_bins,
            f_min=0,
            f_max=sample_rate // 2,
        )
        mel_spec = transform(wave)  # (n_mels, T)
        # 转换为 log scale
        feature = torch.log(mel_spec + 1e-8).transpose(0, 1)  # (T, n_mels)
        return feature.numpy()


def ctc_greedy_decode(log_probs: np.ndarray, blank_id: int = 0) -> list:
    """
    CTC greedy 解码
    
    Args:
        log_probs: shape (batch_size, T, vocab_size) 的 log 概率
        blank_id: blank token ID
    
    Returns:
        解码后的 token ID 列表
    """
    assert log_probs.ndim == 3, log_probs.shape
    batch_size = log_probs.shape[0]
    
    results = []
    for b in range(batch_size):
        # 对每个时间步取 argmax
        pred_ids = log_probs[b].argmax(axis=-1)  # (T,)
        
        # 去除连续重复和 blank
        unique_ids = []
        prev_id = -1
        for token_id in pred_ids:
            if token_id != prev_id and token_id != blank_id:
                unique_ids.append(int(token_id))
            prev_id = token_id
        
        results.append(unique_ids)
    
    return results[0] if batch_size == 1 else results


def process_audio_streaming(
    model: SimpleOnnxModel,
    audio_file: str,
    sample_rate: int = 16000,
    chunk_duration: float = 1.0,
):
    """
    流式处理音频文件
    
    Args:
        model: ONNX 模型
        audio_file: 音频文件路径
        sample_rate: 采样率
        chunk_duration: 每次处理的音频时长（秒）
    """
    print(f"加载音频: {audio_file}")
    wave = load_audio(audio_file, sample_rate)
    print(f"音频长度: {wave.shape[0] / sample_rate:.2f} 秒")
    
    # 提取 fbank 特征
    print("提取 fbank 特征...")
    features = extract_fbank(wave, sample_rate)  # (T, 80)
    print(f"特征形状: {features.shape}")
    
    # 将特征分割成 chunk 进行处理
    # 模型需要输入 shape (batch_size, T, 80)，其中 T = 45
    T = model.T  # 45
    offset = model.decode_chunk_len  # 32 (每次处理的帧数)
    
    # 需要将特征按 T 分割，每次偏移 offset
    all_token_ids = []
    num_frames = features.shape[0]
    
    start_idx = 0
    chunk_idx = 0
    
    while start_idx + T <= num_frames:
        # 取一个 chunk
        chunk_features = features[start_idx:start_idx + T]  # (T, 80)
        chunk_features = chunk_features[np.newaxis, :, :]  # (1, T, 80)
        
        # 推理
        log_probs = model(chunk_features)  # (1, chunk_size, vocab_size)
        
        # CTC 解码
        token_ids = ctc_greedy_decode(log_probs, blank_id=0)
        all_token_ids.extend(token_ids)
        
        print(f"Chunk {chunk_idx}: 帧 {start_idx}-{start_idx+T}, 解码出 {len(token_ids)} 个 tokens")
        
        # 移动到下一个 chunk
        start_idx += offset
        chunk_idx += 1
    
    # 处理剩余的帧（如果需要）
    if start_idx < num_frames:
        remaining = num_frames - start_idx
        if remaining > 0:
            # 用零填充到 T
            chunk_features = np.zeros((1, T, 80), dtype=np.float32)
            chunk_features[0, :remaining] = features[start_idx:]
            log_probs = model(chunk_features)
            token_ids = ctc_greedy_decode(log_probs, blank_id=0)
            all_token_ids.extend(token_ids)
            print(f"最后 chunk: 解码出 {len(token_ids)} 个 tokens")
    
    return all_token_ids


def example_usage():
    """使用示例"""
    parser = argparse.ArgumentParser(description="ONNX 流式 CTC 推理")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="ONNX 模型路径",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="音频文件路径",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        help="tokens.txt 路径（用于将 token ID 转换为文本）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="采样率（默认 16000）",
    )
    
    args = parser.parse_args()
    
    # 1. 加载模型
    print(f"加载模型: {args.model}")
    model = SimpleOnnxModel(args.model)
    print(f"模型配置: T={model.T}, decode_chunk_len={model.decode_chunk_len}")
    
    # 2. 处理音频文件
    token_ids = process_audio_streaming(
        model=model,
        audio_file=args.audio,
        sample_rate=args.sample_rate,
    )
    
    print(f"\n总共解码出 {len(token_ids)} 个 tokens")
    print(f"Token IDs: {token_ids[:50]}...")  # 显示前50个
    
    # 3. 如果提供了 tokens.txt，转换为文本
    if args.tokens:
        print(f"\n加载 token 映射: {args.tokens}")
        id2token = {}
        with open(args.tokens, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    token = parts[0]
                    idx = int(parts[1])
                    
                    # 处理字节级 token
                    if token[:3] == "<0x" and token[-1] == ">":
                        token = int(token[1:-1], base=16)
                        token = token.to_bytes(1, byteorder="little")
                    else:
                        token = token.encode(encoding="utf-8")
                    
                    id2token[idx] = token
        
        # 转换为文本
        text_bytes = b""
        for token_id in token_ids:
            if token_id in id2token:
                text_bytes += id2token[token_id]
        
        try:
            text = text_bytes.decode(encoding="utf-8")
            text = text.replace("▁", " ").strip()
            print(f"\n识别文本: {text}")
        except Exception as e:
            print(f"\n解码文本时出错: {e}")
            print(f"原始字节: {text_bytes[:100]}")


if __name__ == "__main__":
    example_usage()

