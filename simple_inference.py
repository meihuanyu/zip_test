"""
最简单的 zipformer CTC ONNX 推理脚本
使用方法:  python simple_inference.py --onnx-model ./data/model.onnx  --bpe-model data/lang_bpe_500/bpe.model --audio-file ./data/0.wav --target-text "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROFFELS"
"""
import argparse
import logging
from typing import Tuple

import kaldifeat
import sentencepiece as spm
import torch
import torchaudio

import onnxruntime as ort
from ctc_align_beam import AlignConfig, StreamingCTCAligner

logger = logging.getLogger(__name__)


class OnnxModel:
    """ONNX 模型包装类，用于推理"""

    def __init__(self, nn_model: str):
        try:
            session_opts = ort.SessionOptions()
            session_opts.inter_op_num_threads = 1
            session_opts.intra_op_num_threads = 1
            self.model = ort.InferenceSession(
                nn_model,
                sess_options=session_opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"无法加载 ONNX 模型 {nn_model}: {e}")
        
        try:
            meta = self.model.get_modelmeta().custom_metadata_map
            logger.info(f"ONNX 模型元数据: {meta}")
        except:
            logger.info("无法读取 ONNX 模型元数据")

    def __call__(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (N, T, C) float32
          x_lens: (N,) int64
        Returns:
          log_probs: (N, T', vocab_size)
          log_probs_len: (N,)
        """
        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="./model.onnx",
        help="ONNX 模型路径",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="BPE 模型路径",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="要推理的音频文件路径",
    )
    parser.add_argument(
        "--target-text",
        type=str,
        required=True,
        help="要对齐的目标文本（与训练时使用的 BPE 模型对应）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="音频采样率",
    )
    return parser

def print_ctc(ctc_output, blank_id, sp):
    # 调试：显示有意义的帧（非blank概率较高的帧）
    log_probs = ctc_output  # ctc_output 已经是 log_softmax 输出
    probs = log_probs.exp()
    T = probs.shape[1]
    blank_prob = probs[0, :, blank_id]
    
    logger.info(f"CTC 输出形状: {ctc_output.shape}")
    # 统计信息
    non_blank_frames = (blank_prob < 0.5).sum().item()
    logger.info(f"总帧数: {T}, 非blank帧数: {non_blank_frames} (blank概率<0.5)")
    
    # 只显示非blank概率较高的帧
    print("\n有意义的帧 (非blank概率 > 0.1):")
    for t in range(T):
        blank_p = blank_prob[t].item()
        if blank_p < 0.9:  # 只显示非blank概率 > 0.1 的帧
            frame_probs = probs[0, t, :]
            top_k = 3
            top_probs, top_indices = torch.topk(frame_probs, top_k)
            
            items = []
            for idx, prob in zip(top_indices, top_probs):
                p = prob.item()
                token_name = sp.id_to_piece(idx.item())
                items.append(f"{token_name}:{p:.2f}")
            
            line = ", ".join(items)
            print(f"  frame {t:3d}: {line}")

def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    device = torch.device("cpu")
    logger.info(f"使用设备: {device}")

    # 加载 ONNX 模型
    logger.info(f"加载 ONNX 模型: {args.onnx_model}")
    onnx_model = OnnxModel(args.onnx_model)

    # 加载 BPE 模型
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    logger.info(f"加载 BPE 模型: {args.bpe_model}")

    # 获取 blank_id 和 vocab_size
    blank_id = sp.piece_to_id("<blk>")
    vocab_size = sp.get_piece_size()
    logger.info(f"vocab_size: {vocab_size}, blank_id: {blank_id}")
    
    # 加载音频并提取特征
    logger.info(f"加载音频: {args.audio_file}")
    wave, sample_rate = torchaudio.load(args.audio_file)
    assert sample_rate == args.sample_rate, (
        f"期望采样率: {args.sample_rate}, 实际: {sample_rate}"
    )
    wave = wave[0].to(device)  # 使用第一个通道
    
    # 提取 fbank 特征
    logger.info("提取 fbank 特征...")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = args.sample_rate
    opts.mel_opts.num_bins = 80  # 固定为 80
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)
    feature = fbank(wave)  # (T, 80)
    feature = feature.unsqueeze(0)  # (1, T, 80)
    feature_lens = torch.tensor([feature.shape[1]], dtype=torch.int64)

    # 1) 使用 ONNX 模型推理，得到 CTC log 概率
    logger.info(f"特征长度: {feature_lens}")
    log_probs, log_probs_len = onnx_model(feature, feature_lens)
    logger.info(f"CTC 输出长度: {log_probs_len}")
    logger.info(f"CTC 输出形状: {log_probs.shape}")

    # log_probs 已经是 (N, T', vocab_size) 格式
    ctc_output = log_probs  # (1, T', vocab_size)

    # 可选：打印每帧 CTC 最高概率 token，便于调试
    # print_ctc(ctc_output, blank_id, sp)

    # 2) 将目标文本编码成 BPE token 序列
    target_tokens_list = sp.encode(args.target_text)
    if len(target_tokens_list) == 0:
        raise ValueError("target_text 编码后长度为 0，请检查与 BPE 模型是否匹配")

    target_tokens = torch.tensor(
        target_tokens_list,
        dtype=torch.long,
    )

    # 3) 流式 CTC Beam 对齐：按帧推进，实时更新 token_index
    #    这里只是用整段 CTC 输出模拟流式场景，真实麦克风场景下可以逐帧/逐块调用同样的接口。
    #    只取当前有效帧 [0, log_probs_len[0])
    valid_T = int(log_probs_len[0].item())
    log_probs_frame = ctc_output[0, :valid_T, :]  # (T, V)

    align_config = AlignConfig()  # 如需调参可修改其字段
    aligner = StreamingCTCAligner(
        target_tokens=target_tokens,
        blank_id=blank_id,
        config=align_config,
    )

    last_token_index = 0
    for t in range(valid_T):
        last_token_index = aligner.step(log_probs_frame[t])
        # 这里打印的是"流式"到达第 t 帧时的当前对齐位置
        print(f"[流式对齐] 帧 {t:4d}, token_index = {sp.id_to_piece(target_tokens_list[last_token_index])}")

    # 4) 打印最终对齐结果（token index 以及对应的部分文本）
    aligned_text = sp.decode(target_tokens_list[: last_token_index + 1])
    # 当前对齐位置对应的“音素/BPE token”
    current_phone = sp.id_to_piece(target_tokens_list[last_token_index])

    logger.info(
        f"CTC 流式 Beam 对齐结果: token_index={last_token_index}, "
        f"对应文本片段: {aligned_text!r}, 当前音素: {current_phone!r}"
    )
    print(f"[CTC 流式对齐] 最终 token_index = {last_token_index}")
    print(f"[CTC 流式对齐] 对应文本片段: {aligned_text}")
    print(f"[CTC 流式对齐] 当前音素/BPE token: {current_phone}")


if __name__ == "__main__":
    main()

