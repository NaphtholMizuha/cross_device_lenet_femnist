import scipy.stats as stats
# 假设我们有两组样本数据

group1 = [22.81, 9.69, 4.50, 11.6]
group2 = [10.42, 8.64, 3.04, 8.28]

levene = stats.levene(group1, group2)
# 进行双样本t检验，计算p值
t_statistic, p_value = stats.ttest_ind(group1, group2)
# 输出p值
print("p值为：", p_value)
print(levene)