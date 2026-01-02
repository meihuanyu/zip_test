import re
from typing import List, Dict, Tuple, Optional, Union

class StreamingAligner:
    """
    基于流式 Token 匹配的强制对齐器。
    参考 alignment.md 的 "Greedy Matching with Dynamic Anchoring" 策略。
    """
    def __init__(self, target_text: str, token_map_path: str = "tokens.txt"):
        self.id_to_token = self._load_tokens(token_map_path)
        self.target_text = target_text
        self.clean_target, self.word_map = self._prepare_target(target_text)
        self.ptr = 0  # 指向 clean_target 的字符索引
        self.last_word_idx = -1
        
        # 冷却窗口
        self.last_match_time = 0.0
        
    def _load_tokens(self, path: str) -> Dict[int, str]:
        mapping = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            token_id = int(parts[-1])
                            token_str = " ".join(parts[:-1])
                            mapping[token_id] = token_str
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error loading tokens: {e}")
        return mapping

    def _prepare_target(self, text: str) -> Tuple[str, List[int]]:
        matches = list(re.finditer(r'\S+', text))
        clean_parts = []
        word_map = [] 
        
        current_len = 0
        for idx, m in enumerate(matches):
            word_raw = m.group()
            word_norm = re.sub(r'[^\w]', '', word_raw).lower()
            if not word_norm:
                continue
                
            if current_len > 0:
                clean_parts.append(" ")
                word_map.append(-1) 
                current_len += 1
                
            clean_parts.append(word_norm)
            word_map.extend([idx] * len(word_norm))
            current_len += len(word_norm)
            
        clean_target = "".join(clean_parts)
        return clean_target, word_map

    def _normalize_token(self, token_str: str) -> str:
        s = token_str.replace('▁', ' ').replace(' ', ' ')
        return s.lower().strip()

    def process_token(self, token_input: Union[int, str], start: float, end: float = -1.0) -> Optional[Dict]:
        # 1. 获取 Token 字符串
        raw_token = ""
        if isinstance(token_input, int):
            if token_input <= 2: # 忽略特殊 token
                return None
            raw_token = self.id_to_token.get(token_input, "")
        elif isinstance(token_input, str):
            raw_token = token_input
            
        if not raw_token:
            return None
            
        token_str = self._normalize_token(raw_token)
        if not token_str:
            print(f"[Aligner] Skipped empty token: {raw_token}")
            return None

        # 2. 匹配逻辑
        
        # 自动跳过 target 中的空格
        temp_ptr = self.ptr
        while temp_ptr < len(self.clean_target) and self.clean_target[temp_ptr] == ' ':
            temp_ptr += 1
            
        if temp_ptr >= len(self.clean_target):
            print("[Aligner] End of target text reached")
            return None 
            
        # 策略 A: 精确匹配
        if self.clean_target[temp_ptr:].startswith(token_str):
            matched_len = len(token_str)
            new_ptr = temp_ptr + matched_len
            
            match_end_idx = new_ptr - 1
            if match_end_idx < len(self.word_map):
                word_idx = self.word_map[match_end_idx]
                self.ptr = new_ptr
                if word_idx != -1:
                    adjusted_start = max(0.0, start - 0.03) 
                    self.last_word_idx = word_idx
                    return {
                        "index": word_idx,
                        "text": raw_token,
                        "start": adjusted_start,
                        "end": end if end > 0 else start + 0.04
                    }
            return None

        # 策略 B: Lookahead (跳字处理)
        # 只有当 token 足够长时才启用 Lookahead，防止像 'a', 'e' 这种短 token 误匹配导致跳过正常文本
        if len(token_str) > 1:
            search_window = 20
            search_limit = min(len(self.clean_target), temp_ptr + search_window)
            
            found_offset = -1
            for i in range(1, search_window):
                check_pos = temp_ptr + i
                if check_pos >= search_limit:
                    break
                if self.clean_target[check_pos] == ' ':
                    continue
                if self.clean_target[check_pos:].startswith(token_str):
                    found_offset = check_pos
                    break
            
            if found_offset != -1:
                print(f"[Aligner] Lookahead matched: skipped {found_offset - temp_ptr} chars (token: {token_str})")
                new_ptr = found_offset + len(token_str)
                match_end_idx = new_ptr - 1
                if match_end_idx < len(self.word_map):
                    word_idx = self.word_map[match_end_idx]
                    self.ptr = new_ptr
                    if word_idx != -1:
                        adjusted_start = max(0.0, start - 0.03)
                        return {
                            "index": word_idx,
                            "text": raw_token,
                            "start": adjusted_start,
                            "end": end if end > 0 else start + 0.04
                        }

        # 策略 C: 容错
        print(f"[Aligner] Mismatch: token='{token_str}' vs target='{self.clean_target[temp_ptr:temp_ptr+10]}...'")
        return None
