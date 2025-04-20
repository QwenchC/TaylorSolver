import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.mathtext import math_to_image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mfigure
import numpy as np
from taylor_solver import TaylorSolver
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import matplotlib as mpl
import matplotlib.pyplot as plt
import platform
import os
import matplotlib.ticker as ticker

# 1. 首先设置backend
mpl.use('TkAgg')  # 确保在导入pyplot之前设置

# 2. 设置字体和符号显示
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'

# 3. 创建自定义科学记数法格式化器
class MyScalarFormatter(ticker.ScalarFormatter):
    def __call__(self, x, pos=None):
        # 将科学记数法中的'−'替换为'-'
        s = super().__call__(x, pos)
        return s.replace('−', '-')

# 4. 设置默认的格式化器
def configure_axes(ax):
    """配置坐标轴的格式化器"""
    # 设置X和Y轴的格式化
    formatter = MyScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    return ax

# 检查并设置合适的字体
if platform.system() == 'Windows':
    # 检查常见中文字体是否存在
    fonts_to_try = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    font_found = False
    
    # 打印可用字体列表（调试用）
    import matplotlib.font_manager as fm
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"系统可用字体: {system_fonts[:10]}...")  # 只打印前10个避免输出过多
    
    for font in fonts_to_try:
        if font in system_fonts:
            mpl.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            print(f"使用字体: {font}")
            font_found = True
            break
    
    if not font_found:
        print("警告: 未找到合适的中文字体，GUI中的中文可能无法正确显示")
        # 使用系统默认字体
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
else:
    # Linux/Mac系统
    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Hiragino Sans GB']

# 设置数学字体和符号
mpl.rcParams['axes.unicode_minus'] = False  # 关闭Unicode负号
mpl.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern字体
mpl.rcParams['mathtext.default'] = 'it'
mpl.rcParams['font.family'] = 'sans-serif'

# 设置一个标志位，表示不使用外部渲染库
HAS_MATH_TEXT = False

class TaylorSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("泰勒展开公式求解器")
        self.root.geometry("1200x900")
        
        # 配置matplotlib的数学字体设置
        self.configure_math_fonts()
        
        self.solver = TaylorSolver()
        
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架分为左右两部分
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧输入区域
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 右侧框架，包括上部函数预览和下部展开式预览
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # 右侧上部LaTeX显示区域
        right_top_frame = ttk.LabelFrame(right_frame, text="函数公式预览", padding="10")
        right_top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # 创建函数LaTeX预览面板
        self.latex_fig = mfigure.Figure(figsize=(4, 2), dpi=100)
        self.latex_ax = self.latex_fig.add_subplot(111)
        self.latex_canvas = FigureCanvasTkAgg(self.latex_fig, right_top_frame)
        self.latex_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.latex_ax.axis('off')  # 隐藏坐标轴
        
        # 右侧下部展开式预览区域
        right_bottom_frame = ttk.LabelFrame(right_frame, text="泰勒展开结果预览", padding="10")
        right_bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # 创建展开式LaTeX预览面板
        self.expansion_fig = mfigure.Figure(figsize=(4, 4), dpi=100)
        self.expansion_ax = self.expansion_fig.add_subplot(111)
        self.expansion_canvas = FigureCanvasTkAgg(self.expansion_fig, right_bottom_frame)
        self.expansion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.expansion_ax.axis('off')  # 隐藏坐标轴
        
        # 顶部输入框架
        input_frame = ttk.Frame(left_frame, padding="10")
        input_frame.pack(fill=tk.X)
        
        # 函数输入
        ttk.Label(input_frame, text="函数:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.func_var = tk.StringVar()
        self.func_entry = ttk.Entry(input_frame, textvariable=self.func_var, width=30)
        self.func_entry.grid(row=0, column=1, padx=5, pady=5)
        self.func_entry.bind("<KeyRelease>", self.update_latex_preview)
        ttk.Label(input_frame, text="(例如: sin(x), x**2, exp(x))").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 展开点
        ttk.Label(input_frame, text="展开点:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.point_var = tk.StringVar(value="0")
        ttk.Entry(input_frame, textvariable=self.point_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 阶数
        ttk.Label(input_frame, text="展开阶数:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.order_var = tk.StringVar(value="5")
        ttk.Entry(input_frame, textvariable=self.order_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 计算点
        ttk.Label(input_frame, text="计算点:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.eval_var = tk.StringVar(value="1")
        ttk.Entry(input_frame, textvariable=self.eval_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 余项类型
        ttk.Label(input_frame, text="余项类型:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.remainder_type_var = tk.StringVar(value="lagrange")
        remainder_types = ttk.Combobox(input_frame, textvariable=self.remainder_type_var, width=15)
        remainder_types['values'] = ('lagrange', 'cauchy', 'integral', 'peano')
        remainder_types.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(input_frame, text="(拉格朗日、柯西、积分、佩亚诺)").grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 按钮
        button_frame = ttk.Frame(left_frame, padding="10")
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="计算泰勒展开", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="绘制比较图", command=self.plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="余项分析", command=self.analyze_remainder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除", command=self.clear).pack(side=tk.LEFT, padx=5)
        
        # 结果显示区
        result_frame = ttk.LabelFrame(left_frame, text="结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文本结果
        self.result_text = tk.Text(result_frame, height=10, width=80)
        self.result_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 图形结果
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.figure, result_frame)
        self.plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 初始化LaTeX预览
        self.update_latex_preview()
        self.display_expansion_latex("待计算...")
    
    def update_latex_preview(self, event=None):
        """更新LaTeX公式预览"""
        try:
            func_str = self.func_var.get().strip()
            if not func_str:
                self.display_latex("f(x) = ?")
                return
                
            # 尝试解析函数
            expr = parse_expr(func_str)
            latex_str = sp.latex(expr)
            self.display_latex(f"f(x) = {latex_str}")
        except Exception as e:
            self.display_latex("f(x) = ?")
    
    def display_latex(self, latex_str):
        """在面板上显示LaTeX公式，使用固定大小括号并避免Unicode问题"""
        self.latex_ax.clear()
        self.latex_ax.axis('off')
        
        # 处理LaTeX字符串，将自动调整大小的括号替换为固定大小括号
        import re
        fixed_latex = re.sub(r'\\left\(', '(', latex_str)
        fixed_latex = re.sub(r'\\right\)', ')', fixed_latex)
        fixed_latex = re.sub(r'\\left\[', '[', fixed_latex)
        fixed_latex = re.sub(r'\\right\]', ']', fixed_latex)
        
        # 替换可能导致问题的字符
        if "待计算" in fixed_latex:
            fixed_latex = fixed_latex.replace("待计算", "Waiting...")
        fixed_latex = fixed_latex.replace("−", "-")  # 替换Unicode负号
        
        # 使用安全的文本渲染方法
        self.set_safe_text(self.latex_ax, f"${fixed_latex}$", 
                        (0.5, 0.5), fontsize=16, ha='center', va='center',
                        transform=self.latex_ax.transAxes)
        
        self.latex_fig.tight_layout()
        self.latex_canvas.draw()
    
    def display_expansion_latex(self, latex_str, point=None, order=None):
        """显示泰勒展开式，从左到右次数递增，最后加上余项"""
        import re

        # 清除旧的LaTeX预览
        self.expansion_ax.clear()
        self.expansion_ax.axis('off')
        
        # 如果只提供了一个字符串参数，直接显示
        if point is None and order is None:
            self.expansion_ax.text(0.5, 0.5, f"${latex_str}$", 
                                fontsize=14, ha='center', va='center',
                                transform=self.expansion_ax.transAxes)
            self.expansion_fig.tight_layout()
            self.expansion_canvas.draw()
            return
        
        # 无论如何，都重新计算展开式以确保次数递增排列
        if self.solver.function is not None:
            # 手动构建按次数递增排列的展开式的LaTeX字符串
            latex_parts = []
            
            for i in range(order + 1):
                if i == 0:
                    # 常数项
                    term = self.solver.function.subs(self.solver.var, point)
                    if term != 0:
                        latex_parts.append(sp.latex(term))
                else:
                    # i阶项
                    derivative = sp.diff(self.solver.function, self.solver.var, i)
                    coef = derivative.subs(self.solver.var, point) / sp.factorial(i)
                    if coef != 0:
                        # 构建项的LaTeX表示
                        if coef == 1 and i == 1:
                            term_latex = f"(x - {point})"
                        elif coef == 1:
                            term_latex = f"(x - {point})^{{{i}}}"
                        elif coef == -1 and i == 1:
                            term_latex = f"-(x - {point})"
                        elif coef == -1:
                            term_latex = f"-(x - {point})^{{{i}}}"
                        else:
                            # 处理系数为分数的情况
                            coef_latex = sp.latex(coef)
                            if i == 1:
                                term_latex = f"{coef_latex} (x - {point})"
                            else:
                                term_latex = f"{coef_latex} (x - {point})^{{{i}}}"
                        latex_parts.append(term_latex)
            
        # 按照次数递增顺序连接各项
        if latex_parts:
            # 连接所有项，注意处理加减号
            latex_str = latex_parts[0]
            for part in latex_parts[1:]:
                if part.startswith('-'):
                    latex_str += f" {part}"
                else:
                    latex_str += f" + {part}"
        else:
            latex_str = "0"
        
        # 如果展开点是0(或0.0)，简化表达式
        if abs(point) < 1e-10:  # 使用更健壮的方式检查是否为0
            # 处理不同形式的零点表示(0, 0.0等)
            latex_str = re.sub(r'\(x - 0\.?0*\)', 'x', latex_str)
            latex_str = re.sub(r'\(x - 0\.?0*\)\^\{([^}]+)\}', r'x^{\1}', latex_str)
        
        # 根据展开阶数和余项类型，获取对应的余项表达式
        remainder_type = self.remainder_type_var.get()
        _, remainder_expr = self.solver.get_remainder(remainder_type, order, point + 0.1)
        remainder_latex = sp.latex(remainder_expr)
        
        # 处理LaTeX字符串，使用固定大小括号
        import re
        fixed_latex = re.sub(r'\\left\(', '(', latex_str)
        fixed_latex = re.sub(r'\\right\)', ')', latex_str)
        fixed_latex = re.sub(r'\\left\[', '[', latex_str)
        fixed_latex = re.sub(r'\\right\]', ']', latex_str)
        
        # 处理余项LaTeX
        fixed_remainder = re.sub(r'\\left\(', '(', remainder_latex)
        fixed_remainder = re.sub(r'\\right\)', ')', fixed_remainder)
        fixed_remainder = re.sub(r'\\left\[', '[', fixed_remainder)
        fixed_remainder = re.sub(r'\\right\]', ']', fixed_remainder)

        # 处理余项LaTeX
        if remainder_type == "integral":
            # 直接使用简化版的积分表示
            # 1. 首先完全替换\limits命令
            fixed_remainder = remainder_latex.replace('\\limits', '')
            
            # 2. 确保积分上下限格式正确
            fixed_remainder = fixed_remainder.replace('\\int_{', '\\int_{')
            fixed_remainder = fixed_remainder.replace('}^{', '}^{')
            
            # 3. 处理其他可能的格式问题
            fixed_remainder = fixed_remainder.replace('_(', '_{')
            fixed_remainder = fixed_remainder.replace(')^(', ')}^{')
            fixed_remainder = fixed_remainder.replace('\\dt)', '\\, dt')
            
            # 4. 如果所有尝试都失败，使用更简单的表示
            if '\\limits' in fixed_remainder:
                fixed_remainder = f"\\frac{{\\int_{{0}}^{{x}} \\sin(t) \\cdot (x-t)^{{{order}}} \\, dt}}{{({order}!)}}"
        elif remainder_type == "peano":
            # 保持佩亚诺余项的处理不变
            fixed_remainder = re.sub(r'o\s*([^\(].*)', r'o(\1)', fixed_remainder)
            fixed_remainder = re.sub(r'o\s*(x\^\{[^}]+\})', r'o(\1)', fixed_remainder)
        else:
            # 其他余项类型的处理保持不变
            fixed_remainder = re.sub(r'\\left\(', '(', remainder_latex)
            fixed_remainder = re.sub(r'\\right\)', ')', fixed_remainder)
            fixed_remainder = re.sub(r'\\left\[', '[', fixed_remainder)
            fixed_remainder = re.sub(r'\\right\]', ']', fixed_remainder)
        
        # 构建完整的显示内容：T_n(x) = 展开式 + 余项
        full_text = f"T_{{{order}}}(x) = {fixed_latex} + R_{{{order}}}(x)"
        
        # 根据公式长度调整字体大小
        fontsize = 14
        if len(fixed_latex) > 80:
            fontsize = 12
        if len(fixed_latex) > 150:
            fontsize = 10
        
        # 使用安全的文本渲染方法：
        self.set_safe_text(self.expansion_ax, f"${full_text}$", 
                        (0.5, 0.7), fontsize=fontsize, ha='center', va='center',
                        transform=self.expansion_ax.transAxes)

        self.set_safe_text(self.expansion_ax, f"$R_{{{order}}}(x) = {fixed_remainder}$",
                        (0.5, 0.3), fontsize=fontsize-2, ha='center', va='center',
                        transform=self.expansion_ax.transAxes, color='red')
        
        self.expansion_fig.tight_layout()
        self.expansion_canvas.draw()
    
    def calculate(self):
        try:
            # 获取输入
            func_str = self.func_var.get()
            point = float(self.point_var.get())
            order = int(self.order_var.get())
            eval_point = float(self.eval_var.get())
            
            # 设置函数和展开点
            if not self.solver.set_function(func_str):
                messagebox.showerror("错误", "函数表达式有误！")
                return
            
            self.solver.set_expansion_point(point)
            
            # 计算展开式
            expansion = self.solver.get_symbolic_expansion(order)
            numerical = self.solver.get_numerical_expansion(order, eval_point)
            
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"函数: {func_str}\n")
            self.result_text.insert(tk.END, f"在点 x = {point} 的 {order} 阶泰勒展开式:\n")
            self.result_text.insert(tk.END, f"{expansion}\n\n")
            self.result_text.insert(tk.END, f"在 x = {eval_point} 处的数值计算结果: {numerical}\n")
            
            # 显示泰勒展开的LaTeX公式
            expansion_latex = sp.latex(expansion)
            self.display_expansion_latex(expansion_latex, point, order)
            
        except Exception as e:
            messagebox.showerror("计算错误", str(e))
    
    def plot(self):
        try:
            # 获取输入
            func_str = self.func_var.get()
            point = float(self.point_var.get())
            order = int(self.order_var.get())
            
            # 设置函数和展开点
            if not self.solver.set_function(func_str):
                messagebox.showerror("错误", "函数表达式有误！")
                return
            
            self.solver.set_expansion_point(point)
            
            # 清除旧图
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # 获取展开式
            expansion = self.solver.get_symbolic_expansion(order)
            
            # 更新展开式预览
            expansion_latex = sp.latex(expansion)
            self.display_expansion_latex(expansion_latex, point, order)
            
            # 绘图逻辑
            # 删除这两行导入语句
            # import sympy as sp
            # import numpy as np
            
            x_range = (-10, 10)
            points = 1000
            x_vals = np.linspace(x_range[0], x_range[1], points)
            
            # 将sympy表达式转换为可以用numpy计算的函数
            f_lambda = sp.lambdify(self.solver.var, self.solver.function, "numpy")
            taylor_lambda = sp.lambdify(self.solver.var, expansion, "numpy")
            
            try:
                y_orig = f_lambda(x_vals)
                y_taylor = taylor_lambda(x_vals)
                
                ax.plot(x_vals, y_orig, 'b-', label='原函数')
                ax.plot(x_vals, y_taylor, 'r--', label=f'泰勒展开（{order}阶）')
                ax.axvline(x=point, color='g', linestyle=':', label=f'展开点 x={point}')
                ax.grid(True)
                ax.legend()
                ax.set_title(f"函数 {func_str} 在 x={point} 处的泰勒展开")
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
                self.plot_canvas.draw()
            except Exception as e:
                messagebox.showerror("绘图错误", str(e))
            
        except Exception as e:
            messagebox.showerror("计算错误", str(e))
    
    def clear(self):
        self.func_var.set("")
        self.point_var.set("0")
        self.order_var.set("5")
        self.eval_var.set("1")
        self.result_text.delete(1.0, tk.END)
        self.figure.clear()
        self.plot_canvas.draw()
        self.update_latex_preview()
        self.display_expansion_latex("待计算...")
    
    def analyze_remainder(self):
        try:
            # 获取输入
            func_str = self.func_var.get()
            point = float(self.point_var.get())
            order = int(self.order_var.get())
            eval_point = float(self.eval_var.get())
            remainder_type = self.remainder_type_var.get()
            
            # 设置函数和展开点
            if not self.solver.set_function(func_str):
                messagebox.showerror("错误", "函数表达式有误！")
                return
            
            self.solver.set_expansion_point(point)
            
            # 获取展开式并更新预览
            expansion = self.solver.get_symbolic_expansion(order)
            expansion_latex = sp.latex(expansion)
            self.display_expansion_latex(expansion_latex, point, order)
            
            # 创建新窗口进行余项分析
            remainder_window = tk.Toplevel(self.root)
            remainder_window.title("泰勒展开余项分析")
            remainder_window.geometry("800x600")
            
            # 创建notebook进行标签页切换
            notebook = ttk.Notebook(remainder_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 标签页1: 余项上界
            tab1 = ttk.Frame(notebook)
            notebook.add(tab1, text=f"{remainder_type.capitalize()}余项")
            
            remainder_val, remainder_expr = self.solver.get_remainder(remainder_type, order, eval_point)
            
            # 添加余项公式的LaTeX预览
            remainder_frame = ttk.Frame(tab1)
            remainder_frame.pack(fill=tk.X, padx=5, pady=5)
            
            rem_fig = mfigure.Figure(figsize=(6, 2), dpi=100)
            self.configure_figure(rem_fig)
            rem_ax = rem_fig.add_subplot(111)
            rem_canvas = FigureCanvasTkAgg(rem_fig, remainder_frame)
            rem_canvas.get_tk_widget().pack(fill=tk.X)
            rem_ax.axis('off')
            
            # 使用LaTeX显示余项表达式
            remainder_latex = sp.latex(remainder_expr)

            # 处理余项LaTeX，特别是积分余项
            if remainder_type == "integral":
                # 移除\limits命令
                fixed_remainder = remainder_latex.replace('\\limits', '')
                
                # 确保积分上下限格式正确
                fixed_remainder = fixed_remainder.replace('\\int_{', '\\int_{')
                fixed_remainder = fixed_remainder.replace('}^{', '}^{')
                
                # 处理其他可能的格式问题
                fixed_remainder = fixed_remainder.replace('_(', '_{')
                fixed_remainder = fixed_remainder.replace(')^(', ')}^{')
                fixed_remainder = fixed_remainder.replace('\\dt)', '\\, dt')
                
                # 如果所有尝试都失败，使用更简单的表示
                if '\\limits' in fixed_remainder:
                    fixed_remainder = f"\\frac{{\\int_{{0}}^{{x}} \\sin(t) \\cdot (x-t)^{{{order}}} \\, dt}}{{({order}!)}}"
            elif remainder_type == "peano":
                # 保持佩亚诺余项的处理不变
                import re
                fixed_remainder = re.sub(r'o\s*([^\(].*)', r'o(\1)', remainder_latex)
                fixed_remainder = re.sub(r'o\s*(x\^\{[^}]+\})', r'o(\1)', fixed_remainder)
            else:
                # 其他余项类型的处理
                import re
                fixed_remainder = re.sub(r'\\left\(', '(', remainder_latex)
                fixed_remainder = re.sub(r'\\right\)', ')', fixed_remainder)
                fixed_remainder = re.sub(r'\\left\[', '[', fixed_remainder)
                fixed_remainder = re.sub(r'\\right\]', ']', fixed_remainder)

            # 显示处理后的余项表达式
            rem_ax.text(0.5, 0.5, f"$R_{{{order}}}(x) = {fixed_remainder}$",
                       fontsize=14, ha='center', va='center', 
                       transform=rem_ax.transAxes)
            rem_fig.tight_layout()
            rem_canvas.draw()
            
            info_text = tk.Text(tab1, height=8, width=80)
            info_text.pack(fill=tk.X, padx=5, pady=5)
            info_text.insert(tk.END, f"函数: {func_str}\n")
            info_text.insert(tk.END, f"在点 x = {point} 处的 {order} 阶泰勒展开\n\n")
            info_text.insert(tk.END, f"{remainder_type.capitalize()}余项表达式:\n")
            info_text.insert(tk.END, f"{remainder_expr}\n\n")
            
            if remainder_val is not None:
                info_text.insert(tk.END, f"在点 x = {eval_point} 处的余项上界估计: {remainder_val}\n")
                
                # 计算相对误差
                try:
                    true_value = float(self.solver.function.subs(self.solver.var, eval_point).evalf())
                    approx_value = self.solver.get_numerical_expansion(order, eval_point)
                    actual_error = abs(true_value - approx_value)
                    relative_error = abs(actual_error / true_value) * 100 if true_value != 0 else float('inf')
                    
                    info_text.insert(tk.END, f"\n真实值: {true_value}\n")
                    info_text.insert(tk.END, f"近似值: {approx_value}\n")
                    info_text.insert(tk.END, f"实际误差: {actual_error}\n")
                    info_text.insert(tk.END, f"相对误差: {relative_error:.6f}%\n")
                except Exception as e:
                    info_text.insert(tk.END, f"\n计算相对误差时出错: {e}\n")
            else:
                info_text.insert(tk.END, "此类型余项不提供数值估计\n")
            
            # 标签页2: 余项收敛性
            tab2 = ttk.Frame(notebook)
            notebook.add(tab2, text="余项收敛性")
            
            fig1 = plt.Figure(figsize=(6, 4), dpi=100)
            self.configure_figure(fig1)
            ax1 = fig1.add_subplot(111)
            canvas1 = FigureCanvasTkAgg(fig1, tab2)
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 绘制余项收敛性图
            max_order = 15
            orders = range(0, max_order + 1)
            remainder_values = []
            
            for o in orders:
                rem_val, _ = self.solver.get_remainder(remainder_type, o, eval_point)
                if rem_val is not None:
                    remainder_values.append(rem_val)
                else:
                    remainder_values.append(float('nan'))
            
            ax1.semilogy(orders, remainder_values, 'o-', label=f'{remainder_type}余项上界')
            ax1.grid(True)
            ax1.set_xlabel('展开阶数')
            ax1.set_ylabel('余项上界 (对数刻度)')
            ax1.set_title(f"函数 {func_str} 在 x={eval_point} 处的泰勒展开余项收敛性")
            ax1.legend()
            
            # 标签页3: 余项比较
            tab3 = ttk.Frame(notebook)
            notebook.add(tab3, text="余项类型比较")
            
            fig2 = plt.Figure(figsize=(6, 4), dpi=100)
            self.configure_figure(fig2)
            ax2 = fig2.add_subplot(111)
            canvas2 = FigureCanvasTkAgg(fig2, tab3)
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 绘制不同余项类型的比较
            delta = max(1, abs(point) * 0.5)
            x_range = (point - delta, point + delta)
            x_vals = np.linspace(x_range[0], x_range[1], 100)
            
            # 计算不同类型的余项
            lagrange_values = []
            cauchy_values = []
            
            for p in x_vals:
                try:
                    lagrange_val, _ = self.solver.get_lagrange_remainder(order, p)
                    lagrange_values.append(lagrange_val)
                    
                    cauchy_val, _ = self.solver.get_cauchy_remainder(order, p)
                    cauchy_values.append(cauchy_val)
                except:
                    lagrange_values.append(np.nan)
                    cauchy_values.append(np.nan)
            
            ax2.semilogy(x_vals, lagrange_values, 'b-', label='拉格朗日余项')
            ax2.semilogy(x_vals, cauchy_values, 'r--', label='柯西余项')
            ax2.axvline(x=point, color='g', linestyle=':', label=f'展开点 x={point}')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title(f"函数 {func_str} 的 {order} 阶泰勒展开余项比较")
            ax2.set_xlabel('x')
            ax2.set_ylabel('余项上界 (对数刻度)')
            
            canvas2.draw()
            
        except Exception as e:
            messagebox.showerror("分析错误", str(e))

    def configure_math_fonts(self):
        """配置数学公式字体和中文字体支持"""
        import matplotlib as mpl
        import platform
        
        # 设置中文字体支持
        if platform.system() == 'Windows':
            # Windows系统
            font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        else:
            # Linux/Mac系统
            font_list = ['WenQuanYi Micro Hei', 'Hiragino Sans GB', 'STHeiti']
        
        # 设置字体
        mpl.rcParams['font.sans-serif'] = font_list + ['DejaVu Sans', 'Arial']
        mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        mpl.rcParams['font.family'] = 'sans-serif'
        
        # 尝试优化内置LaTeX渲染
        mpl.rcParams['text.usetex'] = False  # 我们使用内置渲染
        mpl.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern字体，最接近TeX
        
        # 其他字体设置
        mpl.rcParams['font.serif'] = ['STSong', 'SimSun', 'DejaVu Serif', 'Times New Roman', 'serif']
        
        # 数学符号默认样式
        mpl.rcParams['mathtext.default'] = 'it'  # 或者 'regular'
        
        # 增加输出DPI以提高分辨率
        mpl.rcParams['figure.dpi'] = 150

    def configure_figure(self, fig):
        """为新创建的Figure应用正确的格式化设置"""
        for ax in fig.get_axes():
            configure_axes(ax)
            
            # 特殊处理对数坐标轴
            if hasattr(ax, 'get_yscale') and ax.get_yscale() == 'log':
                # 创建自定义的格式化函数
                def sci_format_func(x, pos):
                    if x == 0:
                        return '0'
                    # 使用字符串格式化确保使用ASCII字符
                    exp = np.floor(np.log10(abs(x)))
                    if abs(exp) > 3:
                        # 返回科学记数法但保证使用ASCII字符
                        coef = x / 10**exp
                        if abs(coef - 1.0) < 1e-6:
                            return f'10$^{{{int(exp)}}}$'
                        else:
                            return f'{coef:.1f} × 10$^{{{int(exp)}}}$'
                    else:
                        # 返回普通格式
                        return f'{x:.6g}'
                
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_format_func))
        
        return fig

    def set_safe_text(self, ax, text_str, position, **kwargs):
        """安全地设置文本，使用ASCII字符"""
        # 处理中文
        if '待计算' in text_str:
            text_str = text_str.replace('待计算', 'Waiting...')
        
        # 直接显示文本，不尝试转换
        return ax.text(position[0], position[1], text_str, **kwargs)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = TaylorSolverGUI(root)
    root.mainloop()