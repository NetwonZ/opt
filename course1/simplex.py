from scipy.optimize import linprog
import numpy as np

class Simplex:
    def __init__(self, C, A, b, senses=None, is_max=True, M=1e6, tol=1e-9, max_iter=200):
        self.C = np.array(C, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.is_max = is_max
        self.M = float(M)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        m = self.A.shape[0]
        if senses is None:
            self.senses = ["<="] * m
        elif isinstance(senses, str):
            self.senses = [senses] * m
        else:
            self.senses = list(senses)

    def solve(self):
        model = self.Mfunction()
        A_ext = model["A_ext"]
        b_vec = model["b"]
        c_ext = model["c_ext"]
        basis = model["basis"][:]
        artificial_idx = model["artificial_idx"]
        n_original = model["n_original"]

        m, n = A_ext.shape

        status = "unknown"
        message = ""
        iterations = 0

        for iterations in range(1, self.max_iter + 1):
            B = A_ext[:, basis]
            try:
                x_B = np.linalg.solve(B, b_vec)
            except np.linalg.LinAlgError:
                status = "error"
                message = "当前基矩阵奇异，无法继续迭代。"
                break

            if np.any(x_B < -self.tol):
                status = "error"
                message = "出现负基变量，当前基不可行。"
                break

            # 解 B^T * pi = c_B，得到单纯形乘子 pi
            c_B = c_ext[basis]
            pi = np.linalg.solve(B.T, c_B)

            # 约化成本：r_j = c_j - pi * A_j（最大化问题）
            reduced = c_ext - pi @ A_ext
            reduced[basis] = 0.0

            entering = int(np.argmax(reduced))
            if reduced[entering] <= self.tol:
                status = "optimal"
                message = "达到最优性条件。"
                break

            d = np.linalg.solve(B, A_ext[:, entering])
            positive_mask = d > self.tol
            if not np.any(positive_mask):
                status = "unbounded"
                message = "目标函数无界。"
                break

            ratios = np.full(m, np.inf)
            ratios[positive_mask] = x_B[positive_mask] / d[positive_mask]
            leaving_pos = int(np.argmin(ratios))
            basis[leaving_pos] = entering
        else:
            status = "iteration_limit"
            message = "达到最大迭代次数。"

        result = {
            "status": status,
            "success": status == "optimal",
            "message": message,
            "iterations": iterations,
        }

        if status != "optimal":
            return result

        B = A_ext[:, basis]
        x_B = np.linalg.solve(B, b_vec)
        x = np.zeros(n)
        x[basis] = x_B

        # 人工变量仍为正，则原问题不可行。
        if artificial_idx and np.any(x[artificial_idx] > self.tol):
            result.update(
                {
                    "status": "infeasible",
                    "success": False,
                    "message": "最优时人工变量仍大于0，原问题不可行。",
                }
            )
            return result

        x_original = x[:n_original]
        fun = float(np.dot(self.C, x_original))
        if not self.is_max:
            # 若是最小化问题，self.C 即原始最小化系数，直接输出最小值。
            objective_value = fun
        else:
            objective_value = fun

        result.update(
            {
                "x": x_original,
                "fun": objective_value,
                "basis": basis,
                "A_ext": A_ext,
                "b": b_vec,
            }
        )
        return result
    
    def Mfunction(self):
        if self.A.ndim != 2:
            raise ValueError("A 必须是二维矩阵。")

        m, n = self.A.shape
        if self.C.shape[0] != n:
            raise ValueError("C 的维度必须与 A 的列数一致。")
        if self.b.shape[0] != m:
            raise ValueError("b 的维度必须与 A 的行数一致。")
        if len(self.senses) != m:
            raise ValueError("senses 的长度必须与约束条数一致。")

        A = self.A.copy()
        b = self.b.copy()
        senses = self.senses[:]

        # 若 b_i < 0，则整行乘 -1，并翻转不等号方向。
        for i in range(m):
            if b[i] < 0:
                A[i, :] *= -1
                b[i] *= -1
                if senses[i] == "<=":
                    senses[i] = ">="
                elif senses[i] == ">=":
                    senses[i] = "<="

        slack_count = sum(1 for s in senses if s == "<=")
        surplus_count = sum(1 for s in senses if s == ">=")
        artificial_count = sum(1 for s in senses if s in (">=", "="))

        total_vars = n + slack_count + surplus_count + artificial_count
        A_ext = np.zeros((m, total_vars), dtype=float)
        A_ext[:, :n] = A

        c_ext = np.zeros(total_vars, dtype=float)
        if self.is_max:
            c_ext[:n] = self.C
        else:
            # 最小化转最大化：min c^T x 等价于 max (-c)^T x
            c_ext[:n] = -self.C

        basis = []
        artificial_idx = []

        slack_col = n
        surplus_col = n + slack_count
        artificial_col = n + slack_count + surplus_count

        for i, s in enumerate(senses):
            if s == "<=":
                A_ext[i, slack_col] = 1.0
                basis.append(slack_col)
                slack_col += 1
            elif s == ">=":
                A_ext[i, surplus_col] = -1.0
                surplus_col += 1

                A_ext[i, artificial_col] = 1.0
                c_ext[artificial_col] = -self.M
                basis.append(artificial_col)
                artificial_idx.append(artificial_col)
                artificial_col += 1
            elif s == "=":
                A_ext[i, artificial_col] = 1.0
                c_ext[artificial_col] = -self.M
                basis.append(artificial_col)
                artificial_idx.append(artificial_col)
                artificial_col += 1
            else:
                raise ValueError("约束类型仅支持 '<=', '>=', '='。")

        return {
            "A_ext": A_ext,
            "b": b,
            "c_ext": c_ext,
            "basis": basis,
            "artificial_idx": artificial_idx,
            "n_original": n,
            "senses": senses,
        }




def Rundemo():
    # max z = 3x1 + 2x2
    C1 = [3, 2]
    A1 = [[2, 1], [1, 3]]
    b1 = [100, 200]

    # 使用大M法单纯形（本例都是 <=，无需人工变量，但流程兼容）。
    solver = Simplex(C1, A1, b1, senses=["<=", "<="], is_max=True)
    res_big_m = solver.solve()
    if res_big_m["success"]:
        print("[大M法] 最优解为:", res_big_m["x"])
        print("[大M法] 最优值为:", res_big_m["fun"])
    else:
        print("[大M法] 求解失败:", res_big_m["message"])

    # 使用 scipy 校验（linprog 默认最小化，因此目标取负）。
    res1 = linprog([-v for v in C1], A_ub=A1, b_ub=b1)
    if res1.success:
        print("[linprog] 最优解为:", res1.x)
        print("[linprog] 最优值为:", -res1.fun)
    else:
        print("[linprog] 求解失败")
        
        
if __name__ == "__main__":
    Rundemo()