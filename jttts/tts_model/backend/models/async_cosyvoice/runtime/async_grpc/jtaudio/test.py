import torch
import torch.nn as nn
from typing import Optional, List, Union

class SegmentWrapper:
    def __init__(
        self,
        module: nn.Module,
        name: str,
        use_compile: bool = True,
        use_cuda_graph: bool = True,
        compile_kwargs: Optional[dict] = None,
        warmup_iters: int = 3
    ):
        """
        封装一个模型段，可选择性地应用编译和CUDA Graph
        
        参数:
            module: 要封装的PyTorch模块
            name: 段的名称(用于调试)
            use_compile: 是否使用torch.compile
            use_cuda_graph: 是否使用CUDA Graph
            compile_kwargs: 传递给torch.compile的参数
            warmup_iters: 预热迭代次数(用于CUDA Graph捕获)
        """
        self.module = module
        self.name = name
        self.use_compile = use_compile
        self.use_cuda_graph = use_cuda_graph
        self.warmup_iters = warmup_iters
        
        # 默认编译参数
        default_compile_kwargs = {
            'mode': 'reduce-overhead',
            'fullgraph': False,
            'dynamic': False
        }
        self.compile_kwargs = default_compile_kwargs | (compile_kwargs or {})
        
        # 应用编译
        if self.use_compile:
            self.module = torch.compile(self.module, **self.compile_kwargs)
        
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.initialized = False
    
    def initialize(self, example_input: torch.Tensor):
        """初始化CUDA Graph(如果启用)"""
        if not self.use_cuda_graph or not torch.cuda.is_available():
            return
        
        # 预热
        self.module(example_input)
        
        # 创建CUDA Graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        self.static_input = example_input.clone().cuda()
        
        # 捕获graph
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.module(self.static_input)
        
        self.initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播"""
        if self.use_cuda_graph and self.initialized:
            # CUDA Graph执行
            self.static_input.copy_(x)
            self.cuda_graph.replay()
            return self.static_output.clone()
        else:
            # 常规执行
            return self.module(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class MultiSegmentModel:
    def __init__(
        self,
        segments: List[nn.Module],
        segment_names: Optional[List[str]] = None,
        compile_flags: Union[bool, List[bool]] = True,
        cuda_graph_flags: Union[bool, List[bool]] = True,
        compile_kwargs: Optional[Union[dict, List[dict]]] = None,
        warmup_iters: int = 3
    ):
        """
        多段模型封装
        
        参数:
            segments: 模型段的列表
            segment_names: 每段的名称列表
            compile_flags: 每段是否编译(可以是单个bool或列表)
            cuda_graph_flags: 每段是否使用CUDA Graph(可以是单个bool或列表)
            compile_kwargs: 每段的编译参数(可以是单个dict或列表)
            warmup_iters: 预热迭代次数
        """
        self.segments = nn.ModuleList(segments)
        self.segment_names = segment_names or [f"segment_{i}" for i in range(len(segments))]
        self.warmup_iters = warmup_iters
        
        # 处理标志参数(统一为列表形式)
        self.compile_flags = self._unify_param(compile_flags, len(segments), True)
        self.cuda_graph_flags = self._unify_param(cuda_graph_flags, len(segments), True)
        
        # 处理编译参数
        if compile_kwargs is None:
            self.compile_kwargs = [{} for _ in range(len(segments))]
        elif isinstance(compile_kwargs, dict):
            self.compile_kwargs = [compile_kwargs for _ in range(len(segments))]
        else:
            self.compile_kwargs = compile_kwargs
        
        # 封装每个段
        self.wrapped_segments = nn.ModuleList([
            SegmentWrapper(
                module=segment,
                name=name,
                use_compile=compile_flag,
                use_cuda_graph=cuda_graph_flag,
                compile_kwargs=compile_kwargs,
                warmup_iters=warmup_iters
            )
            for segment, name, compile_flag, cuda_graph_flag, compile_kwargs in zip(
                self.segments, self.segment_names, 
                self.compile_flags, self.cuda_graph_flags,
                self.compile_kwargs
            )
        ])
        
        self.initialized = False
    
    def _unify_param(self, param, length, default):
        """将参数统一为列表形式"""
        if isinstance(param, (bool, int)):
            return [param for _ in range(length)]
        elif isinstance(param, list):
            if len(param) != length:
                return [default for _ in range(length)]
            return param
        return [default for _ in range(length)]
    
    def initialize(self, example_input: torch.Tensor):
        """初始化所有段(特别是CUDA Graph)"""
        if self.initialized:
            return
        
        # 运行预热
        x = example_input.clone()
        for _ in range(self.warmup_iters):
            for segment in self.wrapped_segments:
                x = segment(x)
        
        # 为每个段初始化CUDA Graph
        x = example_input.clone()
        for segment in self.wrapped_segments:
            segment.initialize(x)
            x = segment(x)  # 获取下一段的输入形状
        
        self.initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行完整的前向传播"""
        for segment in self.wrapped_segments:
            x = segment(x)
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

# 示例用法
if __name__ == "__main__":
    # 1. 定义模型段
    class SegmentA(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        def forward(self, x):
            return self.relu(self.conv2(self.relu(self.conv1(x))))

    class SegmentB(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        def forward(self, x):
            return self.relu(self.conv2(self.relu(self.conv1(x))))

    class SegmentC(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        def forward(self, x):
            x = self.relu(self.conv2(self.relu(self.conv1(x))))
            return self.avgpool(x).flatten(1)

    # 2. 创建多段模型
    model = MultiSegmentModel(
        segments=[SegmentA(), SegmentB(), SegmentC()],
        segment_names=["segment_a", "segment_b", "segment_c"],
        compile_flags=[True, False, True],  # 中间段不使用编译
        cuda_graph_flags=[True, True, True],  # 所有段使用CUDA Graph
        compile_kwargs=[
            {"mode": "max-autotune"},  # 第一段
            {},                        # 第二段(不使用)
            {"mode": "reduce-overhead"} # 第三段
        ],
        warmup_iters=5
    ).cuda()

    # 3. 初始化(包括预热和CUDA Graph捕获)
    example_input = torch.randn(16, 3, 224, 224).cuda()
    model.initialize(example_input)

    # 4. 运行推理
    input_tensor = torch.randn(16, 3, 224, 224).cuda()
    with torch.no_grad():
        output = model(input_tensor)
    
    print("Output shape:", output.shape)