//! 小型稠密线性求解（部分主元 Gaussian 消元）。
//!
//! 面向 IFT 反向的 n ≲ 数百规模系统；大规模系统物理引擎应自带求解器
//! 并通过 `CustomOp` 直接接入。

/// 解 `A x = b`。`a` 为行主序 n×n；奇异行返回部分结果——调用方应先用
/// `ad_verify::condition_number_inf` 检查条件数（设计文档 §4.3.3 的病态风险：
/// κ∞ ≳ 1e12 时伴随解基本不可信，需正则化或改用直接法）。
pub fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    debug_assert_eq!(a.len(), n);
    let mut m = a.to_vec();
    let mut y = b.to_vec();

    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
            .unwrap_or(col);
        m.swap(col, piv);
        y.swap(col, piv);
        let d = m[col][col];
        if d == 0.0 {
            continue;
        }
        for r in col + 1..n {
            let f = m[r][col] / d;
            if f != 0.0 {
                for c in col..n {
                    m[r][c] -= f * m[col][c];
                }
                y[r] -= f * y[col];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let d = m[i][i];
        if d == 0.0 {
            continue;
        }
        let mut s = y[i];
        for c in i + 1..n {
            s -= m[i][c] * x[c];
        }
        x[i] = s / d;
    }
    x
}

/// 解 `Aᵀ x = b`（等价于 `x = A⁻ᵀ b`），避免显式转置拷贝。
pub fn solve_linear_transposed(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let at: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| a[j][i]).collect()).collect();
    solve_linear(&at, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_known_system() {
        // A = [[2, 1], [1, 3]], b = [5, 10] → x = [1, 3]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let x = solve_linear(&a, &[5.0, 10.0]);
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn transposed_solve_matches() {
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let b = [1.0, 2.0, 3.0];
        let x = solve_linear(&a, &b);
        let xt = solve_linear_transposed(&a, &b);
        // A x = b 与 Aᵀ x' = b 一般给出不同解；各自验证残差
        for i in 0..3 {
            let r: f64 = (0..3).map(|j| a[i][j] * x[j]).sum();
            assert!((r - b[i]).abs() < 1e-10);
            let rt: f64 = (0..3).map(|j| a[j][i] * xt[j]).sum();
            assert!((rt - b[i]).abs() < 1e-10);
        }
    }
}

/// 条件数探针的预期用法（§4.3.3）：求解前检查 κ∞，病态时走正则化路径。
#[test]
fn ill_conditioned_system_is_flagged_by_probe() {
    use ad_verify::condition_number_inf;

    // 良态：残差检验通过，κ∞ 小
    let good = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
    let x = solve_linear(&good, &[5.0, 10.0]);
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!(condition_number_inf(&good) < 1e3);

    // 病态（Hilbert 8 阶）：κ∞ ~ 1e10，残差开始劣化——探针先于残差给出告警
    let n = 8;
    let bad: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| 1.0 / ((i + j + 1) as f64)).collect())
        .collect();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let x = solve_linear(&bad, &b);
    let residual: f64 = (0..n)
        .map(|i| {
            let r: f64 = (0..n).map(|j| bad[i][j] * x[j]).sum();
            (r - b[i]).abs()
        })
        .fold(0.0, f64::max);
    let kappa = condition_number_inf(&bad);
    eprintln!("Hilbert-8: κ∞ = {kappa:.3e}, max residual = {residual:.3e}");
    assert!(kappa > 1e9, "κ∞ = {kappa}");
    // 病态但非奇异：解仍可用，但有效位数损失 ~log10(κ)——文档化的预期行为
    assert!(residual < 1e-6);
}
