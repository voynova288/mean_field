# Unified Berry-geometry framework

核心约定：无论体系是 tMBG、TDBG、ATMG、RLG/hBN、HTG，贝利联络、贝利曲率/plaquette flux 和陈数的数值部分都是同一个问题：

1. 在二维动量网格上拿到波函数数组 `psi[i, j, basis, state]`。
2. 用 `WavefunctionIndex` 标记要计算的 `state` 列的物理含义：能带编号、Chern-basis A/B、flavor、valley、系统名等。
3. 如 `k -> k + G_M` 时基底不是字面周期的，就提供 boundary sewing transform，把边界另一侧的波函数先变到同一个基底规范。
4. 统一调用 FHS 链变量：`U_mu(k) = det[Psi(k)^† Psi(k+mu)] / |det[...]|`（单带时退化为重叠相位），得到离散贝利联络 `arg U_mu`、plaquette flux 和 `C = sum flux / 2π`。

因此体系差异不应写进贝利几何公式里，而应写进“如何产生波函数”和“这些波函数列代表什么指标/标签”。

主要入口：

- `WavefunctionIndex`: 记录所选波函数列的物理指标。
- `compute_lattice_topology`: 计算离散 Berry connection、Berry flux 和 Chern number。
- `compute_link_variables`: 只构造链变量，供体系适配层复用。
- `matrix_sewing_transform` 或自定义 callable: 描述 BZ 边界的规范缝合。
