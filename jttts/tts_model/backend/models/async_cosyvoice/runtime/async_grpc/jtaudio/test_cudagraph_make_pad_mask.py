import torch

class PadMaskGenerator:
    def __init__(self):
        self.cuda_graph = None
        self.example_input = None
        self.static_inputs = None
        self.output = None
        
    def __call__(self, lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
        # 处理CPU张量或首次调用
        if lengths.device.type != 'cuda' or self.cuda_graph is None:
            return self._fallback(lengths, max_len)
            
        # 验证输入批次大小是否匹配
        if lengths.size(0) != self.example_input.size(0):
            return self._fallback(lengths, max_len)
            
        # 更新动态输入
        self.static_inputs[0].copy_(lengths)
        
        # 重放CUDA图
        self.cuda_graph.replay()
        return self.output
    
    def _fallback(self, lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
        batch_size = lengths.size(0)
        max_len = max_len if max_len > 0 else lengths.max().item()
        
        # 检查是否可以捕获CUDA图
        if lengths.device.type == 'cuda' and self.cuda_graph is None:
            self._capture_cuda_graph(batch_size, max_len, lengths.device)
            
        # 标准计算路径
        seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return mask
    
    def _capture_cuda_graph(self, batch_size: int, max_len: int, device: torch.device) -> None:
        # 创建示例输入和输出占位符
        self.example_input = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self.output = torch.empty((batch_size, max_len), dtype=torch.bool, device=device)
        
        # 创建静态输入缓冲区
        self.static_inputs = [
            torch.zeros_like(self.example_input)
        ]
        
        # 捕获CUDA图
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            # 创建一个CUDA Graph记录器
            g = torch.cuda.CUDAGraph()
            
            # 分配工作空间
            seq_range = torch.arange(0, max_len, dtype=torch.int64, device=device)
            seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).contiguous()
            seq_length_expand = self.static_inputs[0].unsqueeze(-1)
            
            # 开始捕获
            g.capture_begin()
            mask = seq_range_expand >= seq_length_expand
            self.output.copy_(mask)
            g.capture_end()
            
        # 保存捕获的CUDA图
        self.cuda_graph = g
        
        # 恢复当前流
        torch.cuda.current_stream().wait_stream(s)

# 使用示例
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """使用CUDA Graph优化的掩码生成函数"""
    # 创建单例实例
    if not hasattr(make_pad_mask, 'generator'):
        make_pad_mask.generator = PadMaskGenerator()
    return make_pad_mask.generator(lengths, max_len)    