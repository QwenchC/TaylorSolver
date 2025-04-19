import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mfigure
import numpy as np
from taylor_solver import TaylorSolver
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

class TaylorSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("泰勒展开公式求解器")
        self.root.geometry("800x600")
        
        self.solver = TaylorSolver()
        
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架分为左右两部分
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧输入区域
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 右侧上部LaTeX显示区域
        right_top_frame = ttk.LabelFrame(main_frame, text="函数公式预览", padding="10")
        right_top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # 创建LaTeX预览面板
        self.latex_fig = mfigure.Figure(figsize=(4, 2), dpi=100)
        self.latex_ax = self.latex_fig.add_subplot(111)
        self.latex_canvas = FigureCanvasTkAgg(self.latex_fig, right_top_frame)
        self.latex_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.latex_ax.axis('off')  # 隐藏坐标轴
        
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
        """在面板上显示LaTeX公式"""
        self.latex_ax.clear()
        self.latex_ax.axis('off')
        self.latex_ax.text(0.5, 0.5, f"${latex_str}$", 
                          fontsize=14, ha='center', va='center',
                          transform=self.latex_ax.transAxes)
        self.latex_fig.tight_layout()
        self.latex_canvas.draw()
    
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
            
            # 绘图逻辑
            import sympy as sp
            import numpy as np
            
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

if __name__ == "__main__":
    root = tk.Tk()
    app = TaylorSolverGUI(root)
    root.mainloop()