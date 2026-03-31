from ast import Not
from operator import is_

from scipy.optimize import linprog
import numpy as np

class Simplex:
    def __init__(self, C, A, b,is_max = False):
        self.C = np.array(C, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.is_max = is_max
        self.B_vars = []  # 基变量索引
        self.N_vars = []  # 非基变量索引
        self.constraint_num = self.A.shape[0]  # 约束数量
        self.variable_num = self.A.shape[1] + self.constraint_num  #总变量个数 原变量+人工变量总个数
        self.Nvariable_num = self.A.shape[1] 
        #约束个数 = 人工变量个数 = 基向量个数
        #非基变量个数 = 原变量个数


    def solve(self):
        if self.is_max:
            self.C = -self.C  # 转化为最小化问题
        #b和A的行数必须匹配
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A的行数必须与b的长度匹配")
        #添加 约束项 个数的人工变量
        m = self.A.shape[0]
        self.A = np.hstack((self.A, np.eye(m)))  # 添加人工变量
        self.C = np.hstack((self.C, np.zeros(m)))  
        self.b = np.hstack((np.zeros(m), self.b))

        #由大M法初始基矩阵为人工变量，其余为非基矩阵
        self.B_vars = list(range(self.A.shape[1] - m, self.A.shape[1]))
        self.N_vars = list(range(self.A.shape[1]- m))

        while not self.is_optimal():
            if self.is_unbounded():
                raise ValueError("问题无界")
            #取出d中小于0的元素索引
            t = np.where(self.d < 0)[0]
            alpha_list = -self.b[t]/self.d[t]
            min_index  = np.argmin(alpha_list)
            alpha = alpha_list[min_index]
            #最小值的索引作为出基变量索引
            self.q = int(t[min_index])
            #更新基变量和非基变量索引
            self.B_vars.remove(self.q)
            self.B_vars.append(self.p)
            self.N_vars.remove(self.p)
            self.N_vars.append(self.q)
            
            #更新b
            self.b = self.b + alpha * self.d
            
        print("最优解为:", self.b[self.B_vars])
        print("最优值为:", self.C[self.B_vars] @ self.b[self.B_vars])

    def is_optimal(self):
    
        """
        计算检验数，检测是否最优
        output:
        """
        self.B_vars = sorted(self.B_vars)
        self.N_vars = sorted(self.N_vars)
        R = np.zeros(self.variable_num)
        for index in self.N_vars:
            # 计算检验数
            B = self.A[:, self.B_vars]
            r = self.C[index] - self.C[self.B_vars]@ np.linalg.inv(B) @ self.A[:, index]
            R[index] = r
        #如果所有检验数都大于等于0，则当前解为最优解
        is_optimal = np.all(R[self.N_vars] >= 0)
        #否则返回检验数最小的非基变量索引，作为入基向量
        if not is_optimal:
            self.p = int(np.where(R<0)[0][0])
        
        return is_optimal
    def is_unbounded(self):
        """
        检测是否无界
        """
        B = self.A[:, self.B_vars]
        #构建d
        
        d_1 = - np.linalg.inv(B) @ self.A[:, self.p]
        d_2 = np.zeros(self.Nvariable_num)
        d_2[self.p] = 1
        #直接拼接 此时的索引顺序为[self.B_vars, self.N_vars]
        d = np.hstack((d_1, d_2))
        #调整d的顺序，使其与原变量索引一致
        d = d[np.argsort(self.B_vars + self.N_vars)]
        self.d = d
        is_unbounded = np.all(d >= 0)

        
        return is_unbounded
        




def Rundemo():
    # max z = -2x1 + x2
    C1 = [-2, 1]
    A1 = [[-1, 1], [2, 1]]
    b1 = [4, 6]

    # 使用大M法单纯形（本例都是 <=，无需人工变量，但流程兼容）。
    solver = Simplex(C1, A1, b1,is_max=True)
    res = solver.solve()


    # 使用 scipy 校验（linprog 默认最小化，因此目标取负）。
    res1 = linprog(c=[-2, 1], A_ub=[[-1, 1], [2, 1]], b_ub=[4, 6], method='highs')
    if res1.success:
        print("[linprog] 最优解为:", res1.x)
        print("[linprog] 最优值为:", res1.fun)
    else:
        print("[linprog] 求解失败")
        
        
if __name__ == "__main__":
    Rundemo()