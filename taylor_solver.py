import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x
from sympy.parsing.sympy_parser import parse_expr

# 设置matplotlib支持中文显示
def set_matplotlib_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 优先使用的中文字体列表
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.family'] = 'sans-serif'  # 使用sans-serif字体族

# 初始化时调用一次
set_matplotlib_chinese_font()

class TaylorSolver:
    def __init__(self):
        self.function = None
        self.point = 0
        self.var = x
    
    def set_function(self, func_str):
        """设置要展开的函数"""
        try:
            self.function = parse_expr(func_str)
            return True
        except Exception as e:
            print(f"函数解析错误: {e}")
            return False
    
    def set_expansion_point(self, point):
        """设置展开点"""
        self.point = point
    
    def get_symbolic_expansion(self, order):
        """获取泰勒展开的符号解析解"""
        if self.function is None:
            raise ValueError("请先设置函数")
        
        expansion = 0
        for i in range(order + 1):
            if i == 0:
                term = self.function.subs(self.var, self.point)
            else:
                derivative = sp.diff(self.function, self.var, i)
                term = derivative.subs(self.var, self.point) * (self.var - self.point)**i / sp.factorial(i)
            expansion += term
        
        return expansion
    
    def get_numerical_expansion(self, order, point):
        """计算泰勒展开在特定点的数值解"""
        expansion = self.get_symbolic_expansion(order)
        return float(expansion.subs(self.var, point).evalf())
    
    def plot_comparison(self, order, x_range=(-10, 10), points=1000):
        """绘制原函数和泰勒展开的比较图"""
        if self.function is None:
            raise ValueError("请先设置函数")
        
        expansion = self.get_symbolic_expansion(order)
        
        x_vals = np.linspace(x_range[0], x_range[1], points)
        
        # 将sympy表达式转换为可以用numpy计算的函数
        f_lambda = sp.lambdify(self.var, self.function, "numpy")
        taylor_lambda = sp.lambdify(self.var, expansion, "numpy")
        
        try:
            y_orig = f_lambda(x_vals)
            y_taylor = taylor_lambda(x_vals)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_orig, 'b-', label='原函数')
            plt.plot(x_vals, y_taylor, 'r--', label=f'泰勒展开（{order}阶）')
            plt.axvline(x=self.point, color='g', linestyle=':', label=f'展开点 x={self.point}')
            plt.grid(True)
            plt.legend()
            plt.title(f"函数 {self.function} 在 x={self.point} 处的泰勒展开")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        except Exception as e:
            print(f"绘图错误: {e}")

    def get_lagrange_remainder(self, order, point):
        """
        计算拉格朗日(Lagrange)余项
        R_n(x) = f^(n+1)(ξ) * (x-a)^(n+1) / (n+1)!
        其中ξ是介于a和point之间的某个点
        
        返回：上界估计和符号表达式
        """
        if self.function is None:
            raise ValueError("请先设置函数")
            
        # 计算n+1阶导数
        derivative = sp.diff(self.function, self.var, order + 1)
        
        # 创建符号表达式
        remainder_expr = derivative * (self.var - self.point)**(order + 1) / sp.factorial(order + 1)
        
        # 计算上界
        # 1. 确定区间[a,point]内高阶导数的最大值
        # 简化处理：取若干采样点，找出最大值
        if self.point == point:  # 如果point与展开点相同，余项为0
            return 0, sp.sympify(0)
        
        samples = 50
        interval = np.linspace(self.point, point, samples)
        
        # 创建导数函数
        derivative_func = sp.lambdify(self.var, derivative, "numpy")
        
        try:
            # 计算各点的导数值的绝对值
            derivative_vals = np.abs(derivative_func(interval))
            max_derivative = np.max(derivative_vals)
            
            # 计算余项上界
            upper_bound = max_derivative * abs(point - self.point)**(order + 1) / sp.factorial(order + 1)
            
            return float(upper_bound), remainder_expr
        except Exception as e:
            print(f"计算余项时出错: {e}")
            return None, remainder_expr

    def get_cauchy_remainder(self, order, point):
        """
        计算柯西(Cauchy)余项
        R_n(x) = f^(n+1)(ξ) * (x-a)^(n+1) / n! * (x-ξ)/(x-a)
        其中ξ是介于a和point之间的某个点
        
        返回：上界估计和符号表达式
        """
        if self.function is None:
            raise ValueError("请先设置函数")
            
        # 计算n+1阶导数
        derivative = sp.diff(self.function, self.var, order + 1)
        
        # 创建符号表达式
        xi = sp.Symbol('ξ')
        remainder_expr = derivative.subs(self.var, xi) * (self.var - self.point)**(order + 1) / sp.factorial(order) * (self.var - xi)/(self.var - self.point)
        
        # 计算上界
        if self.point == point:  # 如果point与展开点相同，余项为0
            return 0, sp.sympify(0)
        
        samples = 50
        interval = np.linspace(self.point, point, samples)
        
        # 创建导数函数
        derivative_func = sp.lambdify(self.var, derivative, "numpy")
        
        try:
            # 计算各点的导数值的绝对值
            derivative_vals = np.abs(derivative_func(interval))
            max_derivative = np.max(derivative_vals)
            
            # 对于柯西余项，最大值会在ξ接近point时出现
            # 假设最坏情况，xi取使|f^(n+1)(ξ)|最大的值，且(x-ξ)取最大值abs(point-self.point)
            upper_bound = max_derivative * abs(point - self.point)**(order + 1) / sp.factorial(order)
            
            return float(upper_bound), remainder_expr
        except Exception as e:
            print(f"计算余项时出错: {e}")
            return None, remainder_expr

    def get_integral_remainder(self, order, point):
        """
        计算积分形式余项
        R_n(x) = 1/n! * ∫_a^x f^(n+1)(t) * (x-t)^n dt
        
        返回：上界估计和符号表达式
        """
        if self.function is None:
            raise ValueError("请先设置函数")
            
        # 计算n+1阶导数
        derivative = sp.diff(self.function, self.var, order + 1)
        
        # 创建符号表达式
        t = sp.Symbol('t')
        integrand = derivative.subs(self.var, t) * (self.var - t)**order
        remainder_expr = sp.Integral(integrand, (t, self.point, self.var)) / sp.factorial(order)
        
        # 计算上界（与拉格朗日余项相同，但更直观地表达为积分）
        if self.point == point:  # 如果point与展开点相同，余项为0
            return 0, remainder_expr
        
        # 使用与拉格朗日余项相同的上界估计
        return self.get_lagrange_remainder(order, point)[0], remainder_expr

    def get_peano_remainder(self, order, point):
        """
        计算佩亚诺型(Peano)余项
        R_n(x) = o((x-a)^n)
        
        返回：一个表示佩亚诺余项的表达式（这个余项主要用于理论分析，不提供数值估计）
        """
        if self.function is None:
            raise ValueError("请先设置函数")
        
        # 佩亚诺余项表述为比(x-a)^n更高阶的无穷小
        remainder_expr = sp.Symbol('o') * (self.var - self.point)**order
        
        # 佩亚诺余项主要用于理论分析，实际计算可以用拉格朗日余项
        # 这里返回0作为占位符，提醒用户这是无法直接计算的
        return None, remainder_expr

    def get_remainder(self, remainder_type, order, point):
        """
        根据指定的余项类型获取余项
        
        参数:
        remainder_type: 字符串，'lagrange', 'cauchy', 'integral', 'peano'之一
        order: 泰勒展开的阶数
        point: 计算余项的点
        
        返回: (数值上界, 符号表达式)
        """
        if remainder_type.lower() == 'lagrange':
            return self.get_lagrange_remainder(order, point)
        elif remainder_type.lower() == 'cauchy':
            return self.get_cauchy_remainder(order, point)
        elif remainder_type.lower() == 'integral':
            return self.get_integral_remainder(order, point)
        elif remainder_type.lower() == 'peano':
            return self.get_peano_remainder(order, point)
        else:
            raise ValueError(f"未知的余项类型: {remainder_type}")

    def plot_remainder_comparison(self, point, order, x_range=None):
        """
        比较不同余项类型的上界
        """
        if self.function is None:
            raise ValueError("请先设置函数")
        
        # 如果未指定范围，使用默认范围
        if x_range is None:
            delta = max(1, abs(self.point) * 0.5)
            x_range = (self.point - delta, self.point + delta)
        
        # 在多个点上计算不同类型的余项
        points = np.linspace(x_range[0], x_range[1], 100)
        lagrange_values = []
        cauchy_values = []
        
        for p in points:
            try:
                lagrange_val, _ = self.get_lagrange_remainder(order, p)
                lagrange_values.append(lagrange_val)
                
                cauchy_val, _ = self.get_cauchy_remainder(order, p)
                cauchy_values.append(cauchy_val)
            except:
                lagrange_values.append(np.nan)
                cauchy_values.append(np.nan)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(points, lagrange_values, 'b-', label='拉格朗日余项')
        plt.semilogy(points, cauchy_values, 'r--', label='柯西余项')
        plt.axvline(x=self.point, color='g', linestyle=':', label=f'展开点 x={self.point}')
        plt.grid(True)
        plt.legend()
        plt.title(f"函数 {self.function} 的 {order} 阶泰勒展开余项比较")
        plt.xlabel('x')
        plt.ylabel('余项上界 (对数刻度)')
        plt.show()

    def get_remainder_bound(self, order, point):
        """
        计算泰勒展开余项的上界
        """
        return self.get_lagrange_remainder(order, point)[0]

    def plot_remainder_convergence(self, point, max_order=15):
        """
        绘制不同阶数下余项大小的收敛情况
        """
        if self.function is None:
            raise ValueError("请先设置函数")
            
        orders = range(0, max_order + 1)
        remainder_values = []
        
        for order in orders:
            remainder = self.get_remainder_bound(order, point)
            if remainder is not None:
                remainder_values.append(remainder)
            else:
                remainder_values.append(float('nan'))
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(orders, remainder_values, 'o-', label='余项上界')
        plt.grid(True)
        plt.xlabel('展开阶数')
        plt.ylabel('余项上界 (对数刻度)')
        plt.title(f"函数 {self.function} 在 x={point} 处的泰勒展开余项收敛性")
        plt.legend()
        plt.show()

    def plot_error_analysis(self, point, max_order=10, x_range=None):
        """
        分析不同阶数下泰勒展开的误差
        """
        if self.function is None:
            raise ValueError("请先设置函数")
        
        # 如果未指定范围，使用默认范围
        if x_range is None:
            # 根据展开点确定合适的范围
            delta = max(1, abs(self.point) * 0.5)
            x_range = (self.point - delta, self.point + delta)
        
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        
        # 计算原函数的准确值
        f_lambda = sp.lambdify(self.var, self.function, "numpy")
        true_values = f_lambda(x_vals)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制不同阶数的误差
        selected_orders = [1, 2, 3, 5, 10] if max_order >= 10 else list(range(1, max_order + 1))
        for order in selected_orders:
            expansion = self.get_symbolic_expansion(order)
            taylor_lambda = sp.lambdify(self.var, expansion, "numpy")
            approximation = taylor_lambda(x_vals)
            
            error = np.abs(approximation - true_values)
            plt.semilogy(x_vals, error, label=f'{order}阶')
        
        plt.grid(True)
        plt.axvline(x=self.point, color='k', linestyle='--', label='展开点')
        plt.legend()
        plt.title(f"函数 {self.function} 在 x={self.point} 处的泰勒展开误差分析")
        plt.xlabel('x')
        plt.ylabel('误差 (对数刻度)')
        plt.show()