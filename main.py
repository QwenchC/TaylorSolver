from taylor_solver import TaylorSolver
import argparse

def main():
    parser = argparse.ArgumentParser(description='泰勒展开公式求解器')
    parser.add_argument('--function', '-f', type=str, required=True, help='要展开的函数，如 "sin(x)"')
    parser.add_argument('--point', '-p', type=float, default=0, help='展开点，默认为0（麦克劳林展开）')
    parser.add_argument('--order', '-o', type=int, default=5, help='展开阶数，默认为5')
    parser.add_argument('--eval', '-e', type=float, help='计算特定点的数值，如果不提供则只返回解析解')
    parser.add_argument('--plot', '-g', action='store_true', help='绘制函数和泰勒展开的比较图')
    parser.add_argument('--range', '-r', type=float, nargs=2, default=[-10, 10], help='绘图的x范围，默认为[-10, 10]')
    
    # 余项相关参数
    parser.add_argument('--remainder', '-R', action='store_true', help='计算余项上界')
    parser.add_argument('--remainder-type', '-rt', type=str, default='lagrange', 
                        choices=['lagrange', 'cauchy', 'integral', 'peano'],
                        help='余项类型: lagrange(拉格朗日), cauchy(柯西), integral(积分), peano(佩亚诺)')
    parser.add_argument('--rem-convergence', '-rc', action='store_true', help='分析余项随阶数的收敛性')
    parser.add_argument('--error-analysis', '-ea', action='store_true', help='分析不同阶数的误差')
    parser.add_argument('--rem-comparison', '-rcp', action='store_true', help='比较不同余项类型')
    parser.add_argument('--max-order', '-mo', type=int, default=10, help='误差分析中的最大阶数，默认为10')
    
    args = parser.parse_args()
    
    solver = TaylorSolver()
    if not solver.set_function(args.function):
        return
    
    solver.set_expansion_point(args.point)
    
    try:
        # 获取解析解
        expansion = solver.get_symbolic_expansion(args.order)
        print("\n泰勒展开式（解析解）:")
        print(f"{expansion}\n")
        
        # 如果需要计算数值
        if args.eval is not None:
            numerical = solver.get_numerical_expansion(args.order, args.eval)
            print(f"在 x = {args.eval} 处的数值解: {numerical}")
            print(f"展开阶数: {args.order}")
            
            # 如果需要计算余项
            if args.remainder:
                remainder_val, remainder_expr = solver.get_remainder(args.remainder_type, args.order, args.eval)
                if remainder_val is not None:
                    print(f"{args.remainder_type.capitalize()}余项上界估计: {remainder_val}")
                    # 计算相对误差百分比
                    true_value = float(solver.function.subs(solver.var, args.eval).evalf())
                    relative_error = abs(remainder_val / true_value) * 100 if true_value != 0 else float('inf')
                    print(f"相对误差上界: {relative_error:.6f}%")
                print(f"{args.remainder_type.capitalize()}余项表达式:")
                print(f"{remainder_expr}")
        
        # 如果需要绘图
        if args.plot:
            solver.plot_comparison(args.order, args.range)
            
        # 分析余项收敛性
        if args.rem_convergence and args.eval is not None:
            solver.plot_remainder_convergence(args.eval, args.max_order)
            
        # 误差分析
        if args.error_analysis:
            eval_point = args.eval if args.eval is not None else args.point + 1
            solver.plot_error_analysis(eval_point, args.max_order, args.range)
            
        # 比较不同余项类型
        if args.rem_comparison and args.eval is not None:
            solver.plot_remainder_comparison(args.eval, args.order, args.range)
            
    except Exception as e:
        print(f"计算错误: {e}")

if __name__ == "__main__":
    main()