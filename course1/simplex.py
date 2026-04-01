from scipy.optimize import linprog
import numpy as np


class Simplex:
    def __init__(
        self,
        C,
        A,
        b,
        is_max=False,
        constraint_types=None,
        big_m=1e6,
        tol=1e-9,
        max_iter=1000,
        verbose=True,
    ):
        """
        大M法单纯形求解器（默认变量非负 x>=0）。

        参数
        - C: 目标函数系数，长度 n
        - A: 约束矩阵，形状 m x n
        - b: 约束右端向量，长度 m
        - is_max: True 表示最大化；内部转成最小化做迭代
        - constraint_types: 每条约束类型，支持 '<='、'>='、'='，默认全 '<='
        - big_m: 大M法的惩罚系数
        - tol: 数值容差
        - max_iter: 最大迭代次数
        - verbose: 是否打印结果
        """
        self.C_raw = np.array(C, dtype=float)
        self.A_raw = np.array(A, dtype=float)
        self.b_raw = np.array(b, dtype=float)
        self.is_max = is_max

        self.big_m = float(big_m)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = verbose

        self.constraint_types = self._normalize_constraint_types(constraint_types)
        self._validate_inputs()

        # solve() 期间填充的内部状态
        self.A_std = None
        self.b_std = None
        self.c_std = None
        self.basis = None
        self.artificial_vars = None
        self.n_original = self.A_raw.shape[1]
        self.iterations = 0

    def _normalize_constraint_types(self, constraint_types):
        """将约束类型归一化为 '<='、'>='、'='。"""
        m = self.A_raw.shape[0] if self.A_raw.ndim == 2 else 0
        if constraint_types is None:
            return ["<="] * m

        mapping = {
            "<=": "<=",
            "=<": "<=",
            "le": "<=",
            ">=": ">=",
            "=>": ">=",
            "ge": ">=",
            "=": "=",
            "eq": "=",
        }

        normalized = []
        for item in constraint_types:
            key = str(item).strip().lower()
            if key not in mapping:
                raise ValueError(f"不支持的约束类型: {item}")
            normalized.append(mapping[key])
        return normalized

    def _validate_inputs(self):
        """输入维度与参数合法性检查。"""
        if self.A_raw.ndim != 2:
            raise ValueError("A 必须是二维矩阵")
        if self.C_raw.ndim != 1:
            raise ValueError("C 必须是一维向量")
        if self.b_raw.ndim != 1:
            raise ValueError("b 必须是一维向量")

        m, n = self.A_raw.shape
        if len(self.C_raw) != n:
            raise ValueError("C 的长度必须等于 A 的列数")
        if len(self.b_raw) != m:
            raise ValueError("b 的长度必须等于 A 的行数")
        if len(self.constraint_types) != m:
            raise ValueError("constraint_types 的长度必须等于约束条数")
        if self.big_m <= 0:
            raise ValueError("big_m 必须为正数")

    def _preprocess_negative_b(self, A, b, constraint_types):
        """
        处理 b_i < 0 的情况：整行乘 -1 并翻转不等号方向。
        - '<=' <-> '>='
        - '=' 保持不变
        """
        A_new = A.copy()
        b_new = b.copy()
        ct_new = list(constraint_types)

        for i in range(len(b_new)):
            if b_new[i] < 0:
                A_new[i, :] *= -1
                b_new[i] *= -1
                if ct_new[i] == "<=":
                    ct_new[i] = ">="
                elif ct_new[i] == ">=":
                    ct_new[i] = "<="
        return A_new, b_new, ct_new

    def _build_standard_form(self):
        """
        构建标准等式形式 A_std x_std = b_std，并生成初始基。

        约束转换
        - '<=': 添加松弛变量 +1（该变量可直接入基）
        - '>=': 添加剩余变量 -1，再添加人工变量 +1（人工变量入基）
        - '=' : 添加人工变量 +1（人工变量入基）

        目标函数（内部统一做最小化）
        - 原变量: min 目标下使用原系数，max 目标下使用 -C
        - 松弛/剩余变量: 系数 0
        - 人工变量: 系数 +M
        """
        A, b, ctype = self._preprocess_negative_b(
            self.A_raw, self.b_raw, self.constraint_types
        )

        if self.is_max:
            c_for_min = -self.C_raw.copy()
        else:
            c_for_min = self.C_raw.copy()

        m, n = A.shape
        le_count = sum(1 for t in ctype if t == "<=")
        ge_count = sum(1 for t in ctype if t == ">=")
        eq_count = sum(1 for t in ctype if t == "=")

        slack_start = n
        surplus_start = slack_start + le_count
        artificial_start = surplus_start + ge_count
        total_vars = n + le_count + ge_count + ge_count + eq_count

        A_std = np.zeros((m, total_vars), dtype=float)
        A_std[:, :n] = A

        c_std = np.zeros(total_vars, dtype=float)
        c_std[:n] = c_for_min

        basis = [None] * m
        artificial_vars = []

        slack_idx = 0
        surplus_idx = 0
        art_idx = 0

        for i, t in enumerate(ctype):
            if t == "<=":
                col = slack_start + slack_idx
                A_std[i, col] = 1.0
                basis[i] = col
                slack_idx += 1
            elif t == ">=":
                col_surplus = surplus_start + surplus_idx
                A_std[i, col_surplus] = -1.0
                surplus_idx += 1

                col_art = artificial_start + art_idx
                A_std[i, col_art] = 1.0
                c_std[col_art] = self.big_m
                artificial_vars.append(col_art)
                basis[i] = col_art
                art_idx += 1
            else:
                col_art = artificial_start + art_idx
                A_std[i, col_art] = 1.0
                c_std[col_art] = self.big_m
                artificial_vars.append(col_art)
                basis[i] = col_art
                art_idx += 1

        self.A_std = A_std
        self.b_std = b.copy()
        self.c_std = c_std
        self.basis = basis
        self.artificial_vars = artificial_vars

    def _choose_entering_variable(self, reduced_costs, non_basis):
        """选择最负检验数对应的非基变量入基。"""
        min_value = 0.0
        entering = None
        for j in non_basis:
            if reduced_costs[j] < min_value - self.tol:
                min_value = reduced_costs[j]
                entering = j
        return entering

    def solve(self):
        """执行大M法单纯形并返回结构化结果。"""
        self._build_standard_form()

        m, total_vars = self.A_std.shape
        basis = list(self.basis)

        status = "optimal"
        success = False
        message = ""

        for it in range(1, self.max_iter + 1):
            self.iterations = it

            non_basis = [j for j in range(total_vars) if j not in basis]
            B = self.A_std[:, basis]
            c_B = self.c_std[basis]

            try:
                # x_B: 当前基变量值
                x_B = np.linalg.solve(B, self.b_std)
                # y: 单纯形乘子，解 B^T y = c_B
                y = np.linalg.solve(B.T, c_B)
            except np.linalg.LinAlgError:
                status = "numerical_issue"
                message = "基矩阵奇异，无法继续迭代"
                break

            reduced_costs = np.full(total_vars, np.inf)
            for j in non_basis:
                reduced_costs[j] = self.c_std[j] - y @ self.A_std[:, j]

            entering = self._choose_entering_variable(reduced_costs, non_basis)
            if entering is None:
                # 所有检验数均非负，达到最优（针对内部最小化问题）。
                x_full = np.zeros(total_vars, dtype=float)
                x_full[basis] = x_B

                art_values = x_full[self.artificial_vars] if self.artificial_vars else np.array([])
                if len(art_values) > 0 and np.any(art_values > self.tol):
                    status = "infeasible"
                    message = "人工变量在最优解中仍为正，原问题不可行"
                    success = False
                else:
                    status = "optimal"
                    success = True
                break

            # d_B = B^{-1} a_entering
            d_B = np.linalg.solve(B, self.A_std[:, entering])

            # 若 d_B <= 0，则无法通过增加入基变量维持可行，且目标可无限下降。
            if np.all(d_B <= self.tol):
                status = "unbounded"
                message = "目标函数无界"
                success = False
                break

            # 最小比值检验，只考虑 d_B > 0 的候选离基行。
            ratios = []
            for i in range(m):
                if d_B[i] > self.tol:
                    ratios.append((x_B[i] / d_B[i], i))

            if len(ratios) == 0:
                status = "unbounded"
                message = "无可行离基变量，问题无界"
                success = False
                break

            min_ratio = min(ratios, key=lambda x: x[0])[0]
            candidate_rows = [row for ratio, row in ratios if abs(ratio - min_ratio) <= self.tol]

            # 简单 Bland 规则打破平局，减少退化循环风险。
            leaving_row = min(candidate_rows, key=lambda r: basis[r])
            basis[leaving_row] = entering
        else:
            status = "max_iter_exceeded"
            message = "超过最大迭代次数"

        result = {
            "status": status,
            "success": success,
            "message": message,
            "iterations": self.iterations,
            "x": None,
            "objective": None,
            "basis": basis,
        }

        if status in {"optimal", "infeasible"}:
            B = self.A_std[:, basis]
            x_B = np.linalg.solve(B, self.b_std)
            x_full = np.zeros(total_vars, dtype=float)
            x_full[basis] = x_B
            x_original = x_full[: self.n_original]

            # 用用户原始目标直接计算，避免符号混乱。
            obj_user = float(self.C_raw @ x_original)
            result["x"] = x_original
            result["objective"] = obj_user

        if self.verbose:
            print(f"[Simplex] status={result['status']}, success={result['success']}")
            if result["x"] is not None:
                print("[Simplex] 最优解为:", result["x"])
                print("[Simplex] 最优值为:", result["objective"])
            if result["message"]:
                print("[Simplex] 说明:", result["message"])

        return result


def run_demo():
    print("\n===== Case 1: 全 <=, b>=0 =====")
    c1 = [-2, 1]
    A1 = [[-1, 1], [2, 1]]
    b1 = [4, 6]
    solver1 = Simplex(c1, A1, b1, is_max=True, constraint_types=["<=", "<="])
    solver1.solve()

    # linprog 校验：max(-2x1+x2) <=> min(2x1-x2)
    lp1 = linprog(c=[2, -1], A_ub=[[-1, 1], [2, 1]], b_ub=[4, 6], method="highs")
    if lp1.success:
        print("[linprog] x =", lp1.x, " objective(max)=", -lp1.fun)
    else:
        print("[linprog] 求解失败")

    print("\n===== Case 2: 含 >= 约束 =====")
    # min z = 2x1 + 3x2
    # s.t. x1 + x2 >= 5
    #      2x1 + x2 >= 7
    c2 = [2, 3]
    A2 = [[1, 1], [2, 1]]
    b2 = [5, 7]
    solver2 = Simplex(c2, A2, b2, constraint_types=[">=", ">="])
    solver2.solve()

    print("\n===== Case 3: 含 b<0（自动翻转） =====")
    # min z = x1 - x2
    # s.t. -x1 + 2x2 <= -4  -> 翻转为 x1 - 2x2 >= 4
    #      x1 - x2 <= 1
    c3 = [1, -1]
    A3 = [[-1, 2], [1, -1]]
    b3 = [-4, 1]
    solver3 = Simplex(c3, A3, b3, constraint_types=["<=", "<="])
    solver3.solve()

    print("\n===== Case 4: 混合 <=, >=, = =====")
    # min z = 4x1 + x2
    # s.t. 3x1 + x2 = 3
    #      4x1 + 3x2 >= 6
    c4 = [4, 1]
    A4 = [[3, 1], [4, 3]]
    b4 = [3, 6]
    solver4 = Simplex(c4, A4, b4, constraint_types=["=", ">="])
    solver4.solve()


if __name__ == "__main__":
    run_demo()