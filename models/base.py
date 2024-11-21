'''
这段代码定义了一个名为 BaseVAE 的抽象基类，用于变分自编码器（VAE）的实现。
它继承自 torch.nn.Module，并定义了一些抽象方法，这些方法需要在具体的子类中实现
'''
from .types_ import *
from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    #super 关键字用于调用父类（超类）的一个方法。它通常用于子类中，以确保父类的初始化方法或其他方法被正确调用
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
    #raise 关键字用于引发一个异常。它可以用于中断程序的正常流程，并将控制权转移到异常处理器。
    '''
    def encode：定义了一个名为 encode 的方法。
    self：指向类的实例，类似于其他编程语言中的 this 关键字。
    input: Tensor：方法的参数 input，类型注解为 Tensor。这意味着 input 应该是一个 Tensor 类型的对象。
    -> List[Tensor]：方法的返回类型注解，表示该方法应该返回一个 Tensor 类型的列表。
    '''
    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    '''
    def sample: 定义了一个名为 sample 的方法。
    self: 指向类的实例，类似于其他编程语言中的 this 关键字。
    batch_size: int: 方法的参数 batch_size，类型注解为 int，表示批处理大小。
    current_device: int: 方法的参数 current_device，类型注解为 int，表示当前设备（例如 GPU 的编号）。
    **kwargs: 允许接受任意数量的关键字参数。
    -> Tensor: 方法的返回类型注解，表示该方法应该返回一个 Tensor 对象。
    '''
    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    '''
    有 @abstractmethod：forward 和 loss_function 方法被标记为抽象方法，
        任何继承 BaseVAE 的子类必须实现这些方法，否则无法实例化子类。
    无 @abstractmethod：forward 和 loss_function 方法变成普通方法，
        子类不需要实现这些方法也可以被实例化，并且可以调用基类中的方法。
    '''
    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass



